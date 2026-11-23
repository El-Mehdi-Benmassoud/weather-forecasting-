# gan_main.py
"""GAN implementation for next‑frame prediction.

- Reuses the data pipeline from ``main.py`` (ImagePairsSource, ProcessImageMap).
- Uses the same ``continuous_iou_loss`` for the generator.
- Adds a U‑Net style generator and a PatchGAN discriminator implemented with Flax.
- Alternates generator and discriminator updates each batch.
- Logs hyper‑parameters, train/val losses, and generator loss to MLflow.
"""

import os
import math
import argparse
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import mlflow

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from typing import Any
import optax
import flax.traverse_util as traverse_util
import grain.python as grain
import orbax.checkpoint as ocp

print(f"JAX running on: {jax.default_backend()}")

# ---------------------------------------------------------------------------
# Argument parsing (mirrors main.py, adds GAN‑specific options)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='GAN next‑frame prediction')
parser.add_argument('--lr', default=0.0002, type=float, help='Learning rate (generator)')
parser.add_argument('--disc_lr', default=0.0002, type=float, help='Learning rate (discriminator)')
parser.add_argument('--weight_decay', default=0.0, type=float, help='Weight decay')
parser.add_argument('--epochs', default=100, type=int, help='Training epochs')
parser.add_argument('--batch_size', default=4, type=int, help='Batch size (reduced for memory)')
parser.add_argument('--num_workers', default=64, type=int, help='Dataloader workers')
parser.add_argument('--lambda_gan', default=0.1, type=float, help='Weight for adversarial loss')
parser.add_argument('--seed', default=42, type=int, help='Random seed')
args, _ = parser.parse_known_args()

# ---------------------------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------------------------
mlflow.set_experiment("GAN-NextFrame")
mlflow.start_run()
mlflow.log_params({
    "lr": args.lr,
    "disc_lr": args.disc_lr,
    "weight_decay": args.weight_decay,
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "lambda_gan": args.lambda_gan,
    "jax_backend": jax.default_backend(),
})
print(f"MLflow run started: {mlflow.active_run().info.run_id}")

# ---------------------------------------------------------------------------
# Data pipeline (copied from main.py)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Model definitions (U‑Net generator, PatchGAN discriminator)
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    features: int
    kernel: int = 3
    stride: int = 1
    use_bn: bool = True
    @nn.compact
    def __call__(self, x, training=True):
        x = nn.Conv(self.features, (self.kernel, self.kernel), self.stride, padding='SAME')(x)
        if self.use_bn:
            x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        return x

class Down(nn.Module):
    features: int
    @nn.compact
    def __call__(self, x, training=True):
        return ConvBlock(self.features, stride=2)(x, training)

class Up(nn.Module):
    features: int
    @nn.compact
    def __call__(self, x, skip, training=True):
        x = nn.ConvTranspose(self.features, (4, 4), (2, 2), padding='SAME')(x)
        x = jnp.concatenate([x, skip], axis=-1)
        x = ConvBlock(self.features)(x, training)
        return x

class UNetGenerator(nn.Module):
    base: int = 64
    @nn.compact
    def __call__(self, x, training=True):
        B, HW, C = x.shape
        H = W = int(math.sqrt(HW))
        x = x.reshape(B, H, W, C)
        d1 = ConvBlock(self.base)(x, training)
        d2 = Down(self.base*2)(d1, training)
        d3 = Down(self.base*4)(d2, training)
        d4 = Down(self.base*8)(d3, training)
        bott = Down(self.base*16)(d4, training)
        u3 = Up(self.base*8)(bott, d4, training)
        u2 = Up(self.base*4)(u3, d3, training)
        u1 = Up(self.base*2)(u2, d2, training)
        u0 = Up(self.base)(u1, d1, training)
        out = nn.Conv(12, (1, 1))(u0)  # 12 channels = 4 frames * 3
        return out.reshape(B, -1)

class PatchGANDiscriminator(nn.Module):
    base: int = 64
    @nn.compact
    def __call__(self, x, training=True):
        B, HW, C = x.shape
        H = W = int(math.sqrt(HW))
        x = x.reshape(B, H, W, C)
        x = ConvBlock(self.base, stride=2, use_bn=False)(x, training)
        x = ConvBlock(self.base*2, stride=2)(x, training)
        x = ConvBlock(self.base*4, stride=2)(x, training)
        x = ConvBlock(self.base*8, stride=1)(x, training)
        logits = nn.Conv(1, (4, 4), padding='SAME')(x)
        return logits

# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def continuous_iou_loss(preds, targets, smooth=1e-6):
    preds = jax.nn.sigmoid(preds)
    t_min = jnp.min(targets)
    t_max = jnp.max(targets)
    tgt = (targets - t_min) / (t_max - t_min + smooth)
    tgt = jnp.clip(tgt, 0.0, 1.0)
    inter = jnp.sum(jnp.minimum(preds, tgt), axis=-1)
    union = jnp.sum(jnp.maximum(preds, tgt), axis=-1)
    iou = (inter + smooth) / (union + smooth)
    return 1.0 - jnp.mean(iou)

def disc_loss(real_logits, fake_logits):
    real = optax.sigmoid_binary_cross_entropy(real_logits, jnp.ones_like(real_logits))
    fake = optax.sigmoid_binary_cross_entropy(fake_logits, jnp.zeros_like(fake_logits))
    return jnp.mean(real + fake)

def gen_adv_loss(fake_logits):
    return jnp.mean(optax.sigmoid_binary_cross_entropy(fake_logits, jnp.ones_like(fake_logits)))

# ---------------------------------------------------------------------------
# Training state containers with Batch Stats
# ---------------------------------------------------------------------------
class TrainStateG(train_state.TrainState):
    batch_stats: Any = None

class TrainStateD(train_state.TrainState):
    batch_stats: Any = None

def create_opt(lr, wd, steps, epochs):
    sched = optax.cosine_decay_schedule(lr, epochs * steps)
    return optax.adamw(sched, wd)

@jax.jit
def d_step(state_d, real_pair, fake_pair, rng):
    def loss_fn(p):
        batch = jnp.concatenate([real_pair, fake_pair], axis=0)
        logits, mut = state_d.apply_fn(
            {'params': p, 'batch_stats': state_d.batch_stats}, 
            batch, 
            training=True, 
            mutable=['batch_stats']
        )
        r, f = jnp.split(logits, 2, axis=0)
        return disc_loss(r, f), mut['batch_stats']
    
    (loss, new_d_stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(state_d.params)
    state_d = state_d.apply_gradients(grads=grads)
    return state_d.replace(batch_stats=new_d_stats)

@jax.jit
def g_step(state_g, state_d, inp, tgt, rng, lam):
    def loss_fn(p):
        gen, mut = state_g.apply_fn(
            {'params': p, 'batch_stats': state_g.batch_stats}, 
            inp, 
            training=True, 
            mutable=['batch_stats']
        )
        recon = continuous_iou_loss(gen, tgt.reshape(tgt.shape[0], -1))
        
        fake_pair = jnp.concatenate([inp, gen.reshape(inp.shape)], axis=-1)
        adv = gen_adv_loss(
            state_d.apply_fn(
                {'params': state_d.params, 'batch_stats': state_d.batch_stats}, 
                fake_pair, 
                training=False
            )
        )
        return recon + lam * adv, (recon, adv, mut['batch_stats'])
    
    (tot, (recon, adv, new_g_stats)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state_g.params)
    state_g = state_g.apply_gradients(grads=grads)
    return state_g.replace(batch_stats=new_g_stats), tot, recon, adv

# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------
def ckpt_mgr(dir='./gan_ckpt'):
    opts = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
    return ocp.CheckpointManager(os.path.abspath(dir), options=opts)

def save_ckpt(mgr, g, d, epoch, gl, dl):
    mgr.save(epoch, args=ocp.args.StandardSave({
        'gen_state': g,
        'disc_state': d,
        'epoch': epoch,
        'gen_loss': gl,
        'disc_loss': dl,
    }))
    mgr.wait_until_finished()

def load_ckpt(mgr, g, d):
    latest = mgr.latest_step()
    if latest is None:
        return g, d, 0, float('inf'), float('inf')
    rest = mgr.restore(latest, args=ocp.args.StandardRestore({
        'gen_state': g,
        'disc_state': d,
        'epoch': 0,
        'gen_loss': float('inf'),
        'disc_loss': float('inf'),
    }))
    return (rest['gen_state'], rest['disc_state'],
            rest['epoch']+1, rest['gen_loss'], rest['disc_loss'])

# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    dataset_path = '/mnt/c/Users/Acer/Desktop/dataset'
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
    all_paths = []
    if os.path.exists(dataset_path):
        all_paths = sorted([
            os.path.join(dataset_path, f) for f in os.listdir(dataset_path)
            if os.path.isfile(os.path.join(dataset_path, f)) and f.lower().endswith(exts)
        ])
    train_len = int((len(all_paths) - 7) * 0.9)
    train_paths = all_paths[:train_len + 7] if len(all_paths) > 7 else []

    def loader(paths, batch):
        if len(paths) <= 7:
            return []
        src = ImagePairsSource(paths)
        sam = grain.IndexSampler(num_records=len(src), num_epochs=1,
                                shard_options=grain.NoSharding(), shuffle=False, seed=42)
        return grain.DataLoader(data_source=src, sampler=sam,
                                operations=[ProcessImageMap(), grain.Batch(batch, drop_remainder=True)],
                                worker_count=0)

    steps = len(train_paths) // args.batch_size if len(train_paths) > 7 else 0
    if steps == 0:
        print('==> Not enough data. Exiting.')
        exit(0)

    rng = jax.random.PRNGKey(args.seed)
    gen = UNetGenerator()
    disc = PatchGANDiscriminator()
    dummy = jnp.ones((1, 224*224, 12))
    
    g_vars = gen.init(rng, dummy)
    d_vars = disc.init(rng, jnp.concatenate([dummy, dummy], axis=-1))
    
    opt_g = create_opt(args.lr, args.weight_decay, steps, args.epochs)
    opt_d = create_opt(args.disc_lr, args.weight_decay, steps, args.epochs)
    
    state_g = TrainStateG.create(apply_fn=gen.apply, params=g_vars['params'], tx=opt_g, batch_stats=g_vars.get('batch_stats', {}))
    state_d = TrainStateD.create(apply_fn=disc.apply, params=d_vars['params'], tx=opt_d, batch_stats=d_vars.get('batch_stats', {}))
    
    mgr = ckpt_mgr()
    state_g, state_d, start_ep, best_g, best_d = load_ckpt(mgr, state_g, state_d)

    for ep in range(start_ep, args.epochs):
        train = loader(train_paths, args.batch_size)
        gen_loss_sum = 0.0
        adv_loss_sum = 0.0
        for i, batch in enumerate(tqdm(train, total=steps)):
            inp = jnp.array(batch['current_image'])
            tgt = jnp.array(batch['next_image'])
            
            # Generate fake samples (training=False so we don't update batch stats during D step prep)
            rng, sub = jax.random.split(rng)
            gen_out = state_g.apply_fn({'params': state_g.params, 'batch_stats': state_g.batch_stats}, inp, training=False)
            
            fake = jnp.concatenate([inp, gen_out.reshape(inp.shape)], axis=-1)
            real = jnp.concatenate([inp, tgt.reshape(inp.shape)], axis=-1)
            
            # Discriminator step
            state_d = d_step(state_d, real, fake, sub)
            
            # Generator step
            rng, sub = jax.random.split(rng)
            state_g, tot, recon, adv = g_step(state_g, state_d, inp, tgt, sub, args.lambda_gan)
            
            gen_loss_sum += recon.item()
            adv_loss_sum += adv.item()
            
        avg_recon = gen_loss_sum / max(1, i+1)
        avg_adv = adv_loss_sum / max(1, i+1)
        mlflow.log_metric('gen_recon_loss', avg_recon, step=ep)
        mlflow.log_metric('gen_adv_loss', avg_adv, step=ep)
        print(f'Epoch {ep}: Recon {avg_recon:.4f} | Adv {avg_adv:.4f}')
        if avg_recon < best_g:
            best_g = avg_recon
            save_ckpt(mgr, state_g, state_d, ep, avg_recon, avg_adv)
            mlflow.log_metric('best_gen_recon', best_g, step=ep)
            
    mlflow.end_run()
    print('Training finished.')
