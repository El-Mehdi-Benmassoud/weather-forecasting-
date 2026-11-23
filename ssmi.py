import os
import re
import math
import shutil
import argparse
from tqdm.auto import tqdm
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import mlflow

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

# JAX Ecosystem: Grain and Orbax
import grain.python as grain
import orbax.checkpoint as ocp

import sys
from absl import flags
try:
    flags.FLAGS(sys.argv, known_only=True)
except Exception:
    pass

print(f"JAX running on: {jax.default_backend()}")

parser = argparse.ArgumentParser(description='S4D Spatiotemporal Sequence Prediction JAX')
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay')
parser.add_argument('--epochs', default=100, type=int, help='Training epochs')
parser.add_argument('--num_workers', default=64, type=int, help='Number of workers to use for dataloader')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
parser.add_argument('--n_layers', default=1, type=int, help='Number of S4 layers')
parser.add_argument('--d_model', default=64, type=int, help='Model dimension')
parser.add_argument('--dropout', default=0.5, type=float, help='Dropout rate')
parser.add_argument('--prenorm', action='store_true', help='Prenorm')
parser.add_argument('--loss', default='iou', choices=['iou', 'pure_iou', 'l1+mse', 'mse', 'l1'],
                    help='Loss function: iou (IoU + MSE, recommended), pure_iou, l1+mse, mse, or l1')
parser.add_argument('--eval_every', default=5, type=int, help='Visual evaluation interval in epochs')
parser.add_argument('--ckpt_dir', default='./ssmi_checkpoint', type=str, help='Checkpoint directory')
parser.add_argument('--eval_dir', default='./ssmi_eval_images', type=str, help='Evaluation images directory')
parser.add_argument('--seed', default=42, type=int, help='Random seed')

args, _ = parser.parse_known_args()


# ---------------------------------------------------------------------------
# S4D Architecture
# ---------------------------------------------------------------------------
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
    d_input: int = 12                     # 4 frames * 3 channels
    d_output: int = 4 * 224 * 224 * 3     # 4 output frames flattened
    d_model: int = 64 
    n_layers: int = 1 
    dropout: float = 0.5
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


# ---------------------------------------------------------------------------
# Data Pipeline
# ---------------------------------------------------------------------------
class ImagePairsSource:
    """Grain data source for handling 4->4 image sequences natively."""
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        # We need 8 consecutive images to form a valid (4 input -> 4 target) sequence
        return max(0, len(self.image_paths) - 7)

    def __getitem__(self, idx):
        return {
            "current": self.image_paths[idx : idx + 4],
            "next": self.image_paths[idx + 4 : idx + 8]
        }


class ProcessImageMap(grain.MapTransform):
    """Grain transform replicating the torchvision pipeline exactly via NumPy/PIL."""
    def process_image(self, path):
        mean = np.array([0.40583992, 0.03281116, 0.78520891], dtype=np.float32)
        std = np.array([0.11256019, 0.05567516, 0.10792409], dtype=np.float32)
        try:
            img = Image.open(path).convert('RGB')
        except (OSError, IOError) as e:
            print(f"Warning: Skipping corrupted image {path}: {e}")
            return (np.zeros((224 * 224, 3), dtype=np.float32) - mean) / std

        w, h = img.size
        left = (w - 224) // 2
        top = (h - 224) // 2
        img = img.crop((left, top, left + 224, top + 224))

        arr = np.array(img, dtype=np.float32) / 255.0
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


def create_dataloader(source_paths, batch_size, num_workers=0, shuffle=False):
    if len(source_paths) <= 7:
        return []

    source = ImagePairsSource(source_paths)
    sampler = grain.IndexSampler(
        num_records=len(source),
        num_epochs=1,
        shard_options=grain.NoSharding(),
        shuffle=shuffle,
        seed=args.seed
    )

    return grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=[
            ProcessImageMap(),
            grain.Batch(batch_size=batch_size, drop_remainder=True)
        ],
        worker_count=num_workers
    )


# ---------------------------------------------------------------------------
# Losses & Training Utilities
# ---------------------------------------------------------------------------
def continuous_iou_loss(preds: jnp.ndarray, targets: jnp.ndarray,
                        smooth: float = 1e-6) -> jnp.ndarray:
    """Continuous Intersection over Union (IoU) Loss based on Jaccard Index."""
    preds = jax.nn.sigmoid(preds)
    t_min = jnp.min(targets)
    t_max = jnp.max(targets)
    targets_norm = jnp.clip((targets - t_min) / (t_max - t_min + smooth), 0.0, 1.0)
    intersection = jnp.sum(jnp.minimum(preds, targets_norm), axis=-1)
    union = jnp.sum(jnp.maximum(preds, targets_norm), axis=-1)
    iou = (intersection + smooth) / (union + smooth)
    return 1.0 - jnp.mean(iou)


def spatiotemporal_loss(preds: jnp.ndarray, targets: jnp.ndarray,
                        loss_type: str = 'iou') -> jnp.ndarray:
    """Reconstruction loss for spatiotemporal sequence prediction matching main.py."""
    if loss_type == 'iou':
        B_ = preds.shape[0]
        return continuous_iou_loss(preds.reshape(B_, -1), targets.reshape(B_, -1)) + jnp.mean((preds - targets) ** 2)
    elif loss_type == 'pure_iou':
        B_ = preds.shape[0]
        return continuous_iou_loss(preds.reshape(B_, -1), targets.reshape(B_, -1))
    elif loss_type == 'l1+mse':
        mse = jnp.mean((preds - targets) ** 2)
        mae = jnp.mean(jnp.abs(preds - targets))
        return mse + 0.1 * mae
    elif loss_type == 'mse':
        return jnp.mean((preds - targets) ** 2)
    elif loss_type == 'l1':
        return jnp.mean(jnp.abs(preds - targets))
    else:
        return jnp.mean((preds - targets) ** 2)


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
    params_key, _ = jax.random.split(rng)
    dummy_input = jnp.ones((1, 224 * 224, d_input)) 
    variables = model.init(params_key, dummy_input, deterministic=True)
    params = variables['params']
    tx = setup_optimizer(params, lr, weight_decay, epochs, steps_per_epoch)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def setup_checkpoint_manager(ckpt_dir='./ssmi_checkpoint'):
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
        try:
            restored = mngr.restore(latest_step, args=ocp.args.StandardRestore(target_dict))
            state = restored['state']
            start_epoch = restored['epoch'] + 1
            best_loss = restored['loss']
            print(f"==> Resuming from epoch {start_epoch} with best loss {best_loss:.4f}")
            return state, start_epoch, best_loss
        except Exception as e:
            print(f"Notice: Checkpoint at step {latest_step} could not be loaded ({type(e).__name__}), starting fresh.")
            try:
                ckpt_dir = os.path.abspath(mngr._directory)
                shutil.rmtree(ckpt_dir)
                os.makedirs(ckpt_dir, exist_ok=True)
                print(f"  Cleared incompatible checkpoint directory: {ckpt_dir}")
            except Exception:
                pass
    print("==> No checkpoint found. Starting from scratch.")
    return state, 0, float('inf')


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
        loss = spatiotemporal_loss(outputs, targets_flat, loss_type=args.loss)
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
    loss = spatiotemporal_loss(outputs, targets_flat, loss_type=args.loss)
    mse = jnp.mean((outputs - targets_flat) ** 2)
    mae = jnp.mean(jnp.abs(outputs - targets_flat))
    return loss, mse, mae


# ---------------------------------------------------------------------------
# Visualization & Qualitative Evaluation (from main.py)
# ---------------------------------------------------------------------------
def denormalize_frame(arr: np.ndarray) -> np.ndarray:
    """Reverse z-score normalisation and convert back to uint8 RGB."""
    mean = np.array([0.40583992, 0.03281116, 0.78520891], dtype=np.float32)
    std  = np.array([0.11256019, 0.05567516, 0.10792409], dtype=np.float32)
    arr  = arr * std + mean
    arr  = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def select_eval_indices(val_paths, n_windows, n_samples=10, img_size=224):
    """Select sequences from val_paths, prioritizing those with active cloud dynamics."""
    if n_windows <= 0:
        return []

    scores = []
    step = max(1, n_windows // 100)
    for idx in range(0, n_windows, step):
        try:
            img = Image.open(val_paths[idx]).convert('RGB')
            w, h = img.size
            crop = np.array(img.crop(((w - img_size) // 2, (h - img_size) // 2, (w + img_size) // 2, (h + img_size) // 2)))
            cloud_px = int((crop[..., 1] > 20).sum())
            scores.append((cloud_px, idx))
        except Exception:
            scores.append((0, idx))

    scores.sort(key=lambda x: x[0], reverse=True)
    active_indices = [idx for px, idx in scores if px > 50]

    if len(active_indices) >= n_samples:
        selected = active_indices[:n_samples]
    else:
        if n_windows <= n_samples:
            selected = list(range(n_windows))
        else:
            selected = [
                int(round(i * (n_windows - 1) / (n_samples - 1)))
                for i in range(n_samples)
            ]
    return sorted(selected)


def visualize_predictions(state, val_paths, epoch, n_samples=10, out_dir='./ssmi_eval_images'):
    """Run inference on validation sequences, reverse normalisation,
    and save individual frames + comparison strips exactly like in main.py.
    """
    epoch_dir = os.path.join(out_dir, f'epoch_{epoch:03d}')
    os.makedirs(epoch_dir, exist_ok=True)

    n_windows = len(val_paths) - 7
    sample_indices = select_eval_indices(val_paths, n_windows, n_samples=n_samples, img_size=224)
    if not sample_indices:
        return

    process = ProcessImageMap()

    for rank, idx in enumerate(sample_indices):
        sample_dir = os.path.join(epoch_dir, f'sample_{rank:02d}')
        os.makedirs(sample_dir, exist_ok=True)

        element = {
            "current": val_paths[idx : idx + 4],
            "next": val_paths[idx + 4 : idx + 8]
        }
        processed = process.map(element)
        current_img = jnp.array(processed["current_image"])[None]  # (1, 224*224, 12)
        next_img    = jnp.array(processed["next_image"])[None]     # (1, 224*224, 12)

        # Run inference
        outputs = state.apply_fn(
            {'params': state.params},
            current_img,
            deterministic=True
        )  # (1, 4 * 224 * 224 * 3)

        # Unflatten to (4, 224, 224, 3)
        input_frames = current_img.reshape(1, 224, 224, 4, 3).transpose(0, 3, 1, 2, 4)[0]
        gt_frames    = next_img.reshape(1, 224, 224, 4, 3).transpose(0, 3, 1, 2, 4)[0]
        pred_frames  = outputs.reshape(1, 224, 224, 4, 3).transpose(0, 3, 1, 2, 4)[0]

        # Convert to numpy and denormalize
        input_rgbs = [denormalize_frame(np.array(input_frames[t])) for t in range(4)]
        gt_rgbs    = [denormalize_frame(np.array(gt_frames[t])) for t in range(4)]
        pred_rgbs  = [denormalize_frame(np.array(pred_frames[t])) for t in range(4)]

        # Save individual input frames
        for t in range(4):
            Image.fromarray(input_rgbs[t]).save(os.path.join(sample_dir, f'frame_{t:02d}_input.png'))

        # Save ground truth and predicted frames
        for t in range(4):
            Image.fromarray(gt_rgbs[t]).save(os.path.join(sample_dir, f'frame_{t+1:02d}_gt.png'))
            Image.fromarray(pred_rgbs[t]).save(os.path.join(sample_dir, f'frame_{t+1:02d}_pred.png'))

        # Save side-by-side comparison strip:
        # Row 1: Input 0..3 | GT 1..4
        # Row 2: Input 0..3 | Pred 1..4
        H, W, _ = input_rgbs[0].shape
        strip = np.zeros((H * 2, W * 8, 3), dtype=np.uint8)
        for t in range(4):
            strip[:H, t * W : (t + 1) * W] = input_rgbs[t]
            strip[H:, t * W : (t + 1) * W] = input_rgbs[t]
            strip[:H, (t + 4) * W : (t + 5) * W] = gt_rgbs[t]
            strip[H:, (t + 4) * W : (t + 5) * W] = pred_rgbs[t]

        Image.fromarray(strip).save(os.path.join(sample_dir, 'comparison_strip.png'))

    try:
        mlflow.log_artifacts(epoch_dir, artifact_path=f'eval_images/epoch_{epoch:03d}')
    except Exception as e:
        print(f"Warning: MLflow logging failed: {e}")

    print(f'  📸 Saved {len(sample_indices)} eval samples to {epoch_dir}')


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Natural sort key for numerical ordering (e.g. 1.jpg, 2.jpg, ... 10.jpg)
    def natural_sort_key(s):
        name = os.path.splitext(os.path.basename(s))[0]
        digits = re.findall(r'\d+', name)
        return int(digits[0]) if digits else name

    dataset_path = '/mnt/c/Users/Acer/Desktop/dataset'
    if not os.path.exists(dataset_path):
        # Fallback for Colab
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
        ], key=natural_sort_key)

    # Split train/val with margin of 7 for the 8-image window
    train_len = int((len(all_image_paths) - 7) * 0.9)
    train_paths = all_image_paths[:train_len + 7] if len(all_image_paths) > 7 else []
    val_paths = all_image_paths[train_len:] if len(all_image_paths) > 7 else []

    steps_per_epoch = (len(train_paths) - 7) // args.batch_size if len(train_paths) > 7 else 0
    val_steps = (len(val_paths) - 7) // args.batch_size if len(val_paths) > 7 else 0

    if steps_per_epoch <= 0:
        print('==> Skipped execution: Provide a valid directory containing at least 8 consecutive images.')
        exit(0)

    # --- MLflow Experiment Setup ---
    mlflow.set_experiment("ConvLSTM-vs-S4D")
    mlflow.start_run(run_name="S4D")
    mlflow.log_params({
        "model": "S4D",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
        "prenorm": args.prenorm,
        "loss": args.loss,
        "eval_every": args.eval_every,
        "jax_backend": jax.default_backend(),
    })
    print(f"MLflow run started: {mlflow.active_run().info.run_id}")

    print('==> Building S4D model..')
    model = S4Model(
        d_input=12,
        d_output=4 * 224 * 224 * 3,
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        prenorm=args.prenorm,
    )

    rng = jax.random.PRNGKey(args.seed)
    state = create_train_state(rng, model, 12, args.lr, args.weight_decay, args.epochs, steps_per_epoch)
    mngr = setup_checkpoint_manager(args.ckpt_dir)
    state, start_epoch, best_loss = load_checkpoint(mngr, state)

    print(f'==> Starting training (loss: {args.loss}, eval_every: {args.eval_every} epochs)..')
    for epoch in range(start_epoch, int(args.epochs)):
        trainloader = create_dataloader(train_paths, args.batch_size, args.num_workers, shuffle=True)

        train_loss = 0.0
        n_train = 0
        pbar = tqdm(trainloader, total=steps_per_epoch, desc=f'Ep {epoch:03d} [train]')
        for batch_idx, batch in enumerate(pbar):
            inputs = jnp.array(batch["current_image"]) 
            targets = jnp.array(batch["next_image"]) 

            state, loss, rng = train_step(state, inputs, targets, rng)
            train_loss += float(loss)
            n_train += 1

            pbar.set_postfix({'loss': f'{train_loss / n_train:.4f}'})

        avg_train_loss = train_loss / max(1, n_train)
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

        # ---- Validation ----
        eval_loss = 0.0
        eval_mse = 0.0
        eval_mae = 0.0
        n_val = 0
        valloader = create_dataloader(val_paths, args.batch_size, args.num_workers, shuffle=False)

        pbar_val = tqdm(valloader, total=val_steps, desc=f'Ep {epoch:03d} [val]')
        for batch_idx, batch in enumerate(pbar_val):
            inputs = jnp.array(batch["current_image"])
            targets = jnp.array(batch["next_image"])

            loss, mse, mae = eval_step(state, inputs, targets)
            eval_loss += float(loss)
            eval_mse += float(mse)
            eval_mae += float(mae)
            n_val += 1

            pbar_val.set_postfix({'loss': f'{eval_loss / n_val:.4f}'})

        avg_loss = eval_loss / max(1, n_val)
        avg_mse  = eval_mse / max(1, n_val)
        avg_mae  = eval_mae / max(1, n_val)
        mlflow.log_metric("val_loss", avg_loss, step=epoch)
        mlflow.log_metric("val_mse", avg_mse, step=epoch)
        mlflow.log_metric("val_mae", avg_mae, step=epoch)
        print(f"Epoch {epoch:03d} | train_loss: {avg_train_loss:.4f} | val_loss: {avg_loss:.4f} | val_mse: {avg_mse:.6f} | val_mae: {avg_mae:.4f}")

        if avg_loss < best_loss:
            print(f'  ✓ Checkpoint saved (Loss improved from {best_loss:.4f} to {avg_loss:.4f})')
            best_loss = avg_loss
            save_checkpoint(mngr, state, epoch, avg_loss)
            mlflow.log_metric("best_val_loss", best_loss, step=epoch)

        # ---- Visual evaluation every 5 epochs (matching main.py) ----
        if (epoch + 1) % args.eval_every == 0:
            visualize_predictions(
                state, val_paths, epoch, n_samples=10, out_dir=args.eval_dir
            )

    mlflow.end_run()
    print('==> Training finished.')

