import os
import math
import argparse
from tqdm.auto import tqdm
from PIL import Image
import numpy as np

# --- 1. TPU INITIALIZATION (CRITICAL FOR COLAB) ---
if 'COLAB_TPU_ADDR' in os.environ:
    import jax.tools.colab_tpu
    jax.tools.colab_tpu.setup_tpu()
    print("Successfully initialized Colab TPU!")

# Prevent JAX from locking all GPU memory so dataloader workers don't OOM crash on spawn
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# JAX, Flax, and Optax for Model and Training
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import flax.traverse_util as traverse_util
import jax.lib.xla_bridge as xla_bridge

# JAX Ecosystem: Grain and Orbax
import grain.python as grain
import orbax.checkpoint as ocp

print(f"JAX running on: {xla_bridge.get_backend().platform}")

parser = argparse.ArgumentParser(description='Convection S4 JAX')
parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay')
parser.add_argument('--epochs', default=100, type=float, help='Training epochs')
parser.add_argument('--num_workers', default=192, type=int, help='Number of workers to use for dataloader')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size (reduced for TPU memory)')
parser.add_argument('--n_layers', default=2, type=int, help='Number of layers (reduced for TPU memory)')
parser.add_argument('--d_model', default=64, type=int, help='Model dimension (reduced for TPU memory)')
parser.add_argument('--dropout', default=0.1, type=float, help='Dropout')
parser.add_argument('--prenorm', action='store_true', help='Prenorm')

args, _ = parser.parse_known_args()

class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""
    d_model: int
    N: int = 64
    dt_min: float = 0.001
    dt_max: float = 0.1

    def setup(self):
        H = self.d_model
        N = self.N

        self.log_dt = self.param(
            'log_dt',
            lambda rng: jax.random.uniform(rng, (H,)) * (math.log(self.dt_max) - math.log(self.dt_min)) + math.log(self.dt_min)
        )

        self.C_real = self.param('C_real', lambda rng: jax.random.normal(rng, (H, N // 2)))
        self.C_imag = self.param('C_imag', lambda rng: jax.random.normal(rng, (H, N // 2)))

        self.log_A_real = self.param('log_A_real', lambda rng: jnp.log(0.5 * jnp.ones((H, N // 2))))
        self.A_imag = self.param('A_imag', lambda rng: math.pi * jnp.tile(jnp.arange(N // 2), (H, 1)))

    def __call__(self, L):
        dt = jnp.exp(self.log_dt) # (H)
        C = self.C_real + 1j * self.C_imag # (H, N/2)
        A = -jnp.exp(self.log_A_real) + 1j * self.A_imag # (H, N/2)

        dtA = A * jnp.expand_dims(dt, -1)  # (H, N/2)
        K = jnp.expand_dims(dtA, -1) * jnp.arange(L) # (H, N/2, L)

        C = C * (jnp.exp(dtA) - 1.) / A
        K_out = 2 * jnp.einsum('hn, hnl -> hl', C, jnp.exp(K)).real

        return K_out

class S4D(nn.Module):
    d_model: int
    d_state: int = 64
    dropout: float = 0.0
    transposed: bool = True

    def setup(self):
        self.h = self.d_model
        self.n = self.d_state

        self.D = self.param('D', lambda rng: jax.random.normal(rng, (self.h,)))
        self.kernel = S4DKernel(d_model=self.h, N=self.n)
        self.activation = nn.gelu

        broadcast = (2,) if self.transposed else (1,)
        self.dropout_layer = nn.Dropout(rate=self.dropout, broadcast_dims=broadcast)
        self.out_dense = nn.Dense(2 * self.h)

    def __call__(self, u, deterministic=False):
        if not self.transposed:
            u = jnp.transpose(u, (0, 2, 1))

        L = u.shape[-1]
        k = self.kernel(L=L) 

        k_f = jnp.fft.rfft(k, n=2*L) 
        u_f = jnp.fft.rfft(u, n=2*L) 
        y = jnp.fft.irfft(u_f * k_f, n=2*L)[..., :L] 

        y = y + u * jnp.expand_dims(self.D, -1)
        y = self.activation(y)
        y = self.dropout_layer(y, deterministic=deterministic)

        y_transposed = jnp.transpose(y, (0, 2, 1))
        y_dense = self.out_dense(y_transposed)
        y_out = y_dense[..., :self.h] * jax.nn.sigmoid(y_dense[..., self.h:])

        if self.transposed:
            y_out = jnp.transpose(y_out, (0, 2, 1))

        return y_out, None

class S4Model(nn.Module):
    d_input: int = 12                     # Default updated for 4 frames (4 * 3)
    d_output: int = 4 * 224 * 224 * 3     # Default updated for 4 output frames
    d_model: int = 64 
    n_layers: int = 2 
    dropout: float = 0.2
    prenorm: bool = False

    @nn.compact
    def __call__(self, x, deterministic=False):
        x = nn.Dense(self.d_model, name='encoder')(x)
        x = jnp.transpose(x, (0, 2, 1))  

        for i in range(self.n_layers):
            z = x
            if self.prenorm:
                z = jnp.transpose(z, (0, 2, 1))
                z = nn.LayerNorm(name=f'norm_{i}')(z)
                z = jnp.transpose(z, (0, 2, 1))

            z, _ = S4D(
                d_model=self.d_model,
                dropout=self.dropout,
                transposed=True,
                name=f's4_{i}'
            )(z, deterministic=deterministic)

            z = nn.Dropout(
                rate=self.dropout,
                broadcast_dims=(2,),
                name=f'drop_{i}'
            )(z, deterministic=deterministic)

            x = z + x

            if not self.prenorm:
                x = jnp.transpose(x, (0, 2, 1))
                x = nn.LayerNorm(name=f'norm_{i}')(x)
                x = jnp.transpose(x, (0, 2, 1))

        x = jnp.transpose(x, (0, 2, 1))
        x = jnp.mean(x, axis=1) 
        x = nn.Dense(self.d_output, name='decoder')(x) 

        return x

class ImagePairsSource:
    """Grain data source for handling 4->4 image sequences natively."""
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        # We need 8 consecutive images to form a valid (4 input -> 4 target) sequence
        return len(self.image_paths) - 7

    def __getitem__(self, idx):
        return {
            "current": self.image_paths[idx : idx + 4],
            "next": self.image_paths[idx + 4 : idx + 8]
        }

class ProcessImageMap(grain.MapTransform):
    """Grain transform replicating the torchvision pipeline exactly via NumPy/PIL."""
    def process_image(self, path):
        img = Image.open(path).convert('RGB')

        w, h = img.size
        left = (w - 224) // 2
        top = (h - 224) // 2
        img = img.crop((left, top, left + 224, top + 224))

        arr = np.array(img, dtype=np.float32) / 255.0
        mean = np.array([0.40583992, 0.03281116, 0.78520891], dtype=np.float32)
        std = np.array([0.11256019, 0.05567516, 0.10792409], dtype=np.float32)
        arr = (arr - mean) / std

        return arr.reshape(-1, 3)

    def map(self, element):
        # Process all 4 images and concatenate along the channel/feature axis (-1)
        current_images = [self.process_image(p) for p in element["current"]]
        next_images = [self.process_image(p) for p in element["next"]]

        return {
            "current_image": np.concatenate(current_images, axis=-1),
            "next_image": np.concatenate(next_images, axis=-1)
        }

# Colab mount logic wrapped in try-except for standard environments
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    pass

dataset_path = '/content/drive/MyDrive/dataset'
if os.path.exists(dataset_path):
    print(f"The dataset directory '{dataset_path}' exists.")
else:
    print(f"Warning: Dataset directory '{dataset_path}' does not exist.")

valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
all_image_paths = []
if os.path.exists(dataset_path):
    all_image_paths = sorted([
        os.path.join(dataset_path, fname)
        for fname in os.listdir(dataset_path)
        if os.path.isfile(os.path.join(dataset_path, fname)) and fname.lower().endswith(valid_exts)
    ])

# Split train/val with margin of 7 for the 8-image window
train_len = int((len(all_image_paths) - 7) * 0.9)
train_paths = all_image_paths[:train_len + 7] if len(all_image_paths) > 7 else []
val_paths = all_image_paths[train_len:] if len(all_image_paths) > 7 else []

def create_dataloader(source_paths, batch_size, num_workers):
    if len(source_paths) <= 7: # Validation condition for 8 image span
        return []

    source = ImagePairsSource(source_paths)
    sampler = grain.IndexSampler(
        num_records=len(source),
        num_epochs=1,
        shard_options=grain.NoSharding(),
        shuffle=False,
        seed=42
    )

    return grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=[
            ProcessImageMap(),
            grain.Batch(batch_size=batch_size, drop_remainder=True)
        ],
        worker_count=0
    )

class TrainState(train_state.TrainState):
    pass

def setup_optimizer(params, lr, weight_decay, epochs, steps_per_epoch):
    scheduler = optax.cosine_decay_schedule(init_value=lr, decay_steps=epochs * steps_per_epoch)
    s4_scheduler = optax.cosine_decay_schedule(init_value=min(0.001, lr), decay_steps=epochs * steps_per_epoch)

    adamw = optax.adamw(learning_rate=scheduler, weight_decay=weight_decay)
    s4_adam = optax.adam(learning_rate=s4_scheduler)

    flat_params = traverse_util.flatten_dict(params)
    labels = {path: 's4' if 'kernel' in path else 'regular' for path in flat_params}

    optimizer = optax.multi_transform(
        {'regular': adamw, 's4': s4_adam},
        traverse_util.unflatten_dict(labels)
    )
    return optimizer

def create_train_state(rng, model, d_input, lr, weight_decay, epochs, steps_per_epoch):
    params_key, dropout_key = jax.random.split(rng)
    # Dummy input represents 4 images flattened and concatenated on the feature axis
    dummy_input = jnp.ones((1, 224 * 224, d_input)) 

    variables = model.init(params_key, dummy_input, deterministic=True)
    params = variables['params']

    tx = setup_optimizer(params, lr, weight_decay, epochs, steps_per_epoch)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def setup_checkpoint_manager(ckpt_dir='./checkpoint'):
    options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
    mngr = ocp.CheckpointManager(os.path.abspath(ckpt_dir), options=options)
    return mngr

def save_checkpoint(mngr, state, epoch, loss):
    save_dict = {'state': state, 'epoch': epoch, 'loss': loss}
    mngr.save(epoch, args=ocp.args.StandardSave(save_dict))
    mngr.wait_until_finished()

def load_checkpoint(mngr, state):
    latest_step = mngr.latest_step()
    if latest_step is not None:
        print(f"==> Restoring checkpoint from step {latest_step}")
        target_dict = {'state': state, 'epoch': 0, 'loss': float('inf')}
        restored = mngr.restore(latest_step, args=ocp.args.StandardRestore(target_dict))

        state = restored['state']
        start_epoch = restored['epoch'] + 1
        best_loss = restored['loss']

        print(f"==> Resuming from epoch {start_epoch} with best loss {best_loss:.3f}")
        return state, start_epoch, best_loss

    print("==> No checkpoint found. Starting from scratch.")
    return state, 0, float('inf')

def soft_iou_loss(preds, targets, smooth=1e-6):
    """
    Computes Soft Intersection over Union (IoU) Loss.
    """
    # 1. Sigmoid to constrain model predictions to [0, 1] for valid IoU logic
    preds = jax.nn.sigmoid(preds)
    
    # 2. Rescale normalized targets back to [0, 1] dynamically 
    # (IoU requires positive values, but inputs were standard-scaled in the dataloader)
    targets_min = jnp.min(targets, axis=-1, keepdims=True)
    targets_max = jnp.max(targets, axis=-1, keepdims=True)
    targets_norm = (targets - targets_min) / (targets_max - targets_min + smooth)
    
    # 3. Compute continuous Soft IoU
    intersection = jnp.sum(preds * targets_norm, axis=-1)
    union = jnp.sum(preds + targets_norm - preds * targets_norm, axis=-1)
    iou = (intersection + smooth) / (union + smooth)
    
    return 1.0 - jnp.mean(iou)

@jax.jit
def train_step(state, inputs, targets, prng_key):
    dropout_key, new_key = jax.random.split(prng_key)

    def loss_fn(params):
        outputs = state.apply_fn(
            {'params': params},
            inputs,
            deterministic=False,
            rngs={'dropout': dropout_key}
        )
        targets_flat = targets.reshape(targets.shape[0], -1)
        # Replaced Mean Absolute Error (MAE) with Soft IoU
        loss = soft_iou_loss(outputs, targets_flat)
        return loss, outputs

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, outputs), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, new_key

@jax.jit
def eval_step(state, inputs, targets):
    outputs = state.apply_fn(
        {'params': state.params},
        inputs,
        deterministic=True
    )
    targets_flat = targets.reshape(targets.shape[0], -1)
    # Replaced Mean Absolute Error (MAE) with Soft IoU
    loss = soft_iou_loss(outputs, targets_flat)
    return loss

print('==> Building model..')
model = S4Model(
    d_input=12,                                 # 4 images * 3 channels
    d_output=4 * 224 * 224 * 3,                 # Flattened outputs
    d_model=args.d_model,
    n_layers=args.n_layers,
    dropout=args.dropout,
    prenorm=args.prenorm,
)

rng = jax.random.PRNGKey(42)

steps_per_epoch = len(train_paths) // args.batch_size if len(train_paths) > 7 else 0 

if steps_per_epoch > 0:
    # Notice d_input=12 is passed properly to initialize the network layout
    state = create_train_state(rng, model, 12, args.lr, args.weight_decay, args.epochs, steps_per_epoch)
    mngr = setup_checkpoint_manager()
    state, start_epoch, best_loss = load_checkpoint(mngr, state)

    print('==> Starting training..')
    for epoch in range(start_epoch, int(args.epochs)):
        trainloader = create_dataloader(train_paths, args.batch_size, args.num_workers)

        train_loss = 0
        pbar = tqdm(trainloader, total=steps_per_epoch)
        for batch_idx, batch in enumerate(pbar):
            inputs = jnp.array(batch["current_image"]) 
            targets = jnp.array(batch["next_image"]) 

            state, loss, rng = train_step(state, inputs, targets, rng)
            train_loss += loss.item()

            pbar.set_description(f'Batch Idx: ({batch_idx}/{steps_per_epoch}) | Loss: {train_loss/(batch_idx+1):.3f}')

        eval_loss = 0
        val_steps = len(val_paths) // args.batch_size if len(val_paths) > 7 else 0 
        valloader = create_dataloader(val_paths, args.batch_size, args.num_workers)

        pbar_val = tqdm(valloader, total=val_steps)
        for batch_idx, batch in enumerate(pbar_val):
            inputs = jnp.array(batch["current_image"])
            targets = jnp.array(batch["next_image"])

            loss = eval_step(state, inputs, targets)
            eval_loss += loss.item()

            pbar_val.set_description(f'Eval Batch: ({batch_idx}/{val_steps}) | Loss: {eval_loss/(batch_idx+1):.3f}')

        avg_loss = eval_loss / max(1, val_steps)
        print(f"Epoch: {epoch}, Average Evaluation Loss: {avg_loss:.3f}")

        if avg_loss < best_loss:
            print(f'Saving model... (Loss improved from {best_loss:.3f} to {avg_loss:.3f})')
            best_loss = avg_loss
            save_checkpoint(mngr, state, epoch, avg_loss)

    print('==> Training finished.')
else:
    print('==> Skipped execution: Provide a valid directory containing at least 8 consecutive images.')