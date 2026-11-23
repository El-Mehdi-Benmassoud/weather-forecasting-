# convlstm_main.py
"""ConvLSTM implementation for next-frame prediction."""
import os
import argparse
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import mlflow

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import grain.python as grain
import orbax.checkpoint as ocp

print(f"JAX running on: {jax.default_backend()}")

parser = argparse.ArgumentParser(description='ConvLSTM next-frame prediction')
parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay')
parser.add_argument('--epochs', default=100, type=int, help='Training epochs')
parser.add_argument('--batch_size', default=4, type=int, help='Batch size (small for GPU memory)')
parser.add_argument('--num_workers', default=64, type=int, help='Dataloader workers')
parser.add_argument('--seed', default=42, type=int, help='Random seed')
args, _ = parser.parse_known_args()

mlflow.set_experiment("ConvLSTM-NextFrame")
mlflow.start_run()
mlflow.log_params({
    "lr": args.lr,
    "weight_decay": args.weight_decay,
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "jax_backend": jax.default_backend(),
})
print(f"MLflow run started: {mlflow.active_run().info.run_id}")

class ImagePairsSource:
    def __init__(self, image_paths):
        self.image_paths = image_paths
    def __len__(self):
        return len(self.image_paths) - 7
    def __getitem__(self, idx):
        return {
            "current": self.image_paths[idx: idx + 4],
            "next": self.image_paths[idx + 4: idx + 8]
        }

class ProcessImageMap(grain.MapTransform):
    def process_image(self, path):
        try:
            img = Image.open(path).convert('RGB')
        except (OSError, IOError) as e:
            print(f"Warning: Skipping corrupted image {path}: {e}")
            return np.zeros((224 * 224, 3), dtype=np.float32)
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
        cur = [self.process_image(p) for p in element["current"]]
        nxt = [self.process_image(p) for p in element["next"]]
        return {
            "current_image": np.concatenate(cur, axis=-1),
            "next_image": np.concatenate(nxt, axis=-1)
        }

class ConvLSTMCell(nn.Module):
    features: int
    kernel_size: tuple = (3, 3)

    @nn.compact
    def __call__(self, carry, x):
        c, h = carry
        # x is (B, H, W, C), h is (B, H, W, F)
        inputs = jnp.concatenate([x, h], axis=-1)
        gates = nn.Conv(self.features * 4, self.kernel_size, padding='SAME')(inputs)
        
        i, f, o, g = jnp.split(gates, 4, axis=-1)
        
        i = nn.sigmoid(i)
        f = nn.sigmoid(f)
        o = nn.sigmoid(o)
        g = nn.tanh(g)
        
        c_new = f * c + i * g
        h_new = o * nn.tanh(c_new)
        
        return (c_new, h_new), h_new

class ConvLSTMModel(nn.Module):
    features: int = 16
    out_channels: int = 3
    
    @nn.compact
    def __call__(self, x):
        # x is (B, 224*224, 12)
        B = x.shape[0]
        x = x.reshape(B, 224, 224, 4, 3)
        x = jnp.transpose(x, (0, 3, 1, 2, 4)) # (B, T=4, H=224, W=224, C=3)
        
        T = x.shape[1]
        
        # Initialize hidden states
        c = jnp.zeros((B, 224, 224, self.features))
        h = jnp.zeros((B, 224, 224, self.features))
        carry = (c, h)
        
        cell = ConvLSTMCell(features=self.features)
        
        # Unroll over time
        outputs = []
        for t in range(T):
            carry, h_t = cell(carry, x[:, t])
            # Process output state to match expected output channel size
            out_t = nn.Conv(self.out_channels, (3, 3), padding='SAME')(h_t)
            outputs.append(out_t)
            
        # Stack over time axis
        outputs = jnp.stack(outputs, axis=1) # (B, T=4, H=224, W=224, C=3)
        
        # Reshape to match target flattening (B, 224*224, 12)
        outputs = jnp.transpose(outputs, (0, 2, 3, 1, 4)) # (B, H=224, W=224, T=4, C=3)
        outputs = outputs.reshape(B, -1)
        
        return outputs

def continuous_iou_loss(preds, targets, smooth=1e-6):
    preds = jax.nn.sigmoid(preds)
    t_min = jnp.min(targets)
    t_max = jnp.max(targets)
    targets_norm = (targets - t_min) / (t_max - t_min + smooth)
    targets_norm = jnp.clip(targets_norm, 0.0, 1.0)
    
    intersection = jnp.sum(jnp.minimum(preds, targets_norm), axis=-1)
    union = jnp.sum(jnp.maximum(preds, targets_norm), axis=-1)
    
    iou = (intersection + smooth) / (union + smooth)
    return 1.0 - jnp.mean(iou)

class TrainState(train_state.TrainState):
    pass

def setup_optimizer(lr, weight_decay, epochs, steps_per_epoch):
    scheduler = optax.cosine_decay_schedule(init_value=lr, decay_steps=epochs * steps_per_epoch)
    return optax.adamw(learning_rate=scheduler, weight_decay=weight_decay)

def setup_checkpoint_manager(ckpt_dir='./convlstm_checkpoint'):
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
        target_dict = {'state': state, 'epoch': 0, 'loss': float('inf')}
        restored = mngr.restore(latest_step, args=ocp.args.StandardRestore(target_dict))
        return restored['state'], restored['epoch'] + 1, restored['loss']
    return state, 0, float('inf')

@jax.jit
def train_step(state, inputs, targets, prng_key):
    def loss_fn(params):
        outputs = state.apply_fn({'params': params}, inputs)
        targets_flat = targets.reshape(targets.shape[0], -1)
        loss = continuous_iou_loss(outputs, targets_flat)
        return loss, outputs
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, outputs), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def eval_step(state, inputs, targets):
    outputs = state.apply_fn({'params': state.params}, inputs)
    targets_flat = targets.reshape(targets.shape[0], -1)
    loss = continuous_iou_loss(outputs, targets_flat)
    return loss

if __name__ == '__main__':
    dataset_path = '/mnt/c/Users/Acer/Desktop/dataset'
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
    all_image_paths = []
    if os.path.exists(dataset_path):
        all_image_paths = sorted([
            os.path.join(dataset_path, fname)
            for fname in os.listdir(dataset_path)
            if os.path.isfile(os.path.join(dataset_path, fname)) and fname.lower().endswith(valid_exts)
        ])

    train_len = int((len(all_image_paths) - 7) * 0.9)
    train_paths = all_image_paths[:train_len + 7] if len(all_image_paths) > 7 else []
    val_paths = all_image_paths[train_len:] if len(all_image_paths) > 7 else []

    def create_dataloader(source_paths, batch_size):
        if len(source_paths) <= 7:
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
        
    steps_per_epoch = len(train_paths) // args.batch_size if len(train_paths) > 7 else 0 
    if steps_per_epoch > 0:
        model = ConvLSTMModel()
        rng = jax.random.PRNGKey(args.seed)
        dummy_input = jnp.ones((1, 224 * 224, 12)) 
        
        variables = model.init(rng, dummy_input)
        params = variables['params']
        
        tx = setup_optimizer(args.lr, args.weight_decay, args.epochs, steps_per_epoch)
        state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
        
        mngr = setup_checkpoint_manager()
        state, start_epoch, best_loss = load_checkpoint(mngr, state)

        print('==> Starting training..')
        for epoch in range(start_epoch, int(args.epochs)):
            trainloader = create_dataloader(train_paths, args.batch_size)

            train_loss = 0
            pbar = tqdm(trainloader, total=steps_per_epoch)
            for batch_idx, batch in enumerate(pbar):
                inputs = jnp.array(batch["current_image"]) 
                targets = jnp.array(batch["next_image"]) 
                
                rng, step_rng = jax.random.split(rng)
                state, loss = train_step(state, inputs, targets, step_rng)
                train_loss += loss.item()

                pbar.set_description(f'Batch Idx: ({batch_idx}/{steps_per_epoch}) | Loss: {train_loss/(batch_idx+1):.3f}')

            avg_train_loss = train_loss / max(1, batch_idx + 1)
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

            eval_loss = 0
            val_steps = len(val_paths) // args.batch_size if len(val_paths) > 7 else 0 
            valloader = create_dataloader(val_paths, args.batch_size)

            pbar_val = tqdm(valloader, total=val_steps)
            for batch_idx, batch in enumerate(pbar_val):
                inputs = jnp.array(batch["current_image"])
                targets = jnp.array(batch["next_image"])

                loss = eval_step(state, inputs, targets)
                eval_loss += loss.item()

                pbar_val.set_description(f'Eval Batch: ({batch_idx}/{val_steps}) | Loss: {eval_loss/(batch_idx+1):.3f}')

            avg_loss = eval_loss / max(1, val_steps)
            mlflow.log_metric("val_loss", avg_loss, step=epoch)
            print(f"Epoch: {epoch}, Average Evaluation Loss: {avg_loss:.3f}")

            if avg_loss < best_loss:
                print(f'Saving model... (Loss improved from {best_loss:.3f} to {avg_loss:.3f})')
                best_loss = avg_loss
                save_checkpoint(mngr, state, epoch, avg_loss)
                mlflow.log_metric("best_val_loss", best_loss, step=epoch)

        mlflow.end_run()
        print('==> Training finished.')
    else:
        print('==> Skipped execution: Provide a valid directory containing at least 8 consecutive images.')
