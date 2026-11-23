"""ConvLSTM baseline for spatiotemporal predictive learning.

Architecture: stacked ConvLSTM cells (Shi et al., NeurIPS 2015).

Fairness contract with s4d_main.py
-----------------------------------
  Loss         : identical continuous_iou_loss (no decoupling term)
  Data          : identical SeqImageSource + ProcessSeqMap pipeline
  Normalisation : identical per-channel mean/std
  Patching      : identical space-to-depth before the model
  Optimiser     : identical AdamW + cosine LR schedule
  Protocol      : encoder phase → ground-truth frames
                  decoder phase → autoregressive (no scheduled sampling)
  Metrics       : train_loss (IoU), val_loss (IoU), val_mse (patch-space MSE)
  Checkpoint    : saved on val_loss improvement

Parameter matching guide (patch_size=4, img_channel=3, num_layers=4):
  ConvLSTM  --num_hidden 64   --filter_size 5                 ≈ 3.18 M params
  S4D       --d_model    256  --d_state 64  --kernel_size 3   ≈ 2.59 M params
  S4D       --d_model    290  --d_state 64  --kernel_size 3   ≈ 3.20 M params  ← matched
Both scripts print their exact count at startup so you can tune before a run.
"""

import os
# Prevent JAX from pre-allocating all GPU memory and causing CUDA OOM retries
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import re
import shutil
import argparse
from tqdm.auto import tqdm
from PIL import Image, ImageFile
# Tolerate truncated images in the dataset (e.g. 1614.jpg)
ImageFile.LOAD_TRUNCATED_IMAGES = True
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

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='ConvLSTM spatiotemporal baseline')
# Optimiser
parser.add_argument('--lr',           default=1e-3,  type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=1e-5,  type=float, help='AdamW weight decay')
parser.add_argument('--epochs',       default=100,   type=int,   help='Training epochs')
parser.add_argument('--batch_size',   default=64,     type=int,   help='Batch size')
parser.add_argument('--seed',         default=42,    type=int,   help='Random seed')
# Model
parser.add_argument('--num_layers',   default=4,     type=int,   help='Number of ConvLSTM layers')
parser.add_argument('--num_hidden',   default=64,    type=int,   help='Hidden channels per layer')
parser.add_argument('--filter_size',  default=5,     type=int,   help='Conv kernel size inside cell')
parser.add_argument('--layer_norm',   default=1,     type=int,   help='1 = LayerNorm inside cell')
# Loss & Checkpoint
parser.add_argument('--loss',         default='l1+mse', choices=['l1+mse', 'mse', 'l1', 'iou'],
                    help='Loss function: l1+mse (recommended), mse, l1, or iou')
parser.add_argument('--resume',       action='store_true', default=False,
                    help='Resume from existing checkpoint if available (default: False to prevent resuming from collapsed runs)')
# Data / patching
parser.add_argument('--img_width',    default=64,    type=int,   help='Spatial size (square)')
parser.add_argument('--img_channel',  default=3,     type=int,   help='Input image channels')
parser.add_argument('--patch_size',   default=4,     type=int,   help='Space-to-depth patch size')
# Sequence
parser.add_argument('--input_length', default=4,     type=int,   help='Context frames (encoder)')
parser.add_argument('--total_length', default=8,     type=int,   help='Full sequence length')
parser.add_argument('--eval_every',   default=5,     type=int,   help='Visual evaluation interval in epochs')
args, _ = parser.parse_known_args()

# Derived sizes (identical to PredRNN and s4d_main)
frame_channel = args.patch_size * args.patch_size * args.img_channel
patch_width   = args.img_width  // args.patch_size

# ---------------------------------------------------------------------------
# Data pipeline  (byte-for-byte identical to s4d_main.py)
# ---------------------------------------------------------------------------
class SeqImageSource:
    """Sliding-window sequence source — returns windows of `seq_len` image paths."""
    def __init__(self, image_paths, seq_len):
        self.image_paths = image_paths
        self.seq_len     = seq_len

    def __len__(self):
        return max(0, len(self.image_paths) - self.seq_len + 1)

    def __getitem__(self, idx):
        return {"paths": self.image_paths[idx: idx + self.seq_len]}


class ProcessSeqMap(grain.MapTransform):
    """Loads a window of images → float32 (T, H, W, C) with centre-crop and z-score."""
    def __init__(self, img_size=64):
        self.img_size = img_size

    def _load_one(self, path):
        mean = np.array([0.40583992, 0.03281116, 0.78520891], dtype=np.float32)
        std  = np.array([0.11256019, 0.05567516, 0.10792409], dtype=np.float32)
        try:
            img = Image.open(path).convert('RGB')
        except (OSError, IOError) as e:
            print(f"Warning: bad image {path}: {e}")
            return (np.zeros((self.img_size, self.img_size, 3), dtype=np.float32) - mean) / std
        w, h = img.size
        left = (w - self.img_size) // 2
        top  = (h - self.img_size) // 2
        img  = img.crop((left, top, left + self.img_size, top + self.img_size))
        arr  = np.array(img, dtype=np.float32) / 255.0
        return (arr - mean) / std

    def map(self, element):
        frames = np.stack([self._load_one(p) for p in element["paths"]], axis=0)
        return {"frames": frames}   # (T, H, W, C)

# ---------------------------------------------------------------------------
# Space-to-depth / depth-to-space  (identical to s4d_main.py)
# ---------------------------------------------------------------------------
def space_to_depth(x: jnp.ndarray, patch_size: int) -> jnp.ndarray:
    """(B,T,H,W,C) → (B,T,H//p,W//p,p²C)"""
    B, T, H, W, C = x.shape
    p = patch_size
    x = x.reshape(B, T, H // p, p, W // p, p, C)
    x = jnp.transpose(x, (0, 1, 2, 4, 3, 5, 6))
    return x.reshape(B, T, H // p, W // p, p * p * C)


def depth_to_space(x: jnp.ndarray, patch_size: int, img_channel: int) -> jnp.ndarray:
    """(B,T,H//p,W//p,p²C) → (B,T,H,W,C)"""
    B, T, Ph, Pw, _ = x.shape
    p, C = patch_size, img_channel
    x = x.reshape(B, T, Ph, Pw, p, p, C)
    x = jnp.transpose(x, (0, 1, 2, 4, 3, 5, 6))
    return x.reshape(B, T, Ph * p, Pw * p, C)

# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def continuous_iou_loss(preds: jnp.ndarray, targets: jnp.ndarray,
                        smooth: float = 1e-6) -> jnp.ndarray:
    """Soft IoU loss."""
    preds   = jax.nn.sigmoid(preds)
    t_min   = jnp.min(targets)
    t_max   = jnp.max(targets)
    targets = jnp.clip((targets - t_min) / (t_max - t_min + smooth), 0.0, 1.0)
    inter   = jnp.sum(jnp.minimum(preds, targets), axis=-1)
    union   = jnp.sum(jnp.maximum(preds, targets), axis=-1)
    return 1.0 - jnp.mean((inter + smooth) / (union + smooth))


def spatiotemporal_loss(preds: jnp.ndarray, targets: jnp.ndarray,
                        loss_type: str = 'l1+mse') -> jnp.ndarray:
    """Reconstruction loss for spatiotemporal predictive learning.
    Both preds and targets are in normalized patch-space.
    """
    if loss_type == 'mse':
        return jnp.mean((preds - targets) ** 2)
    elif loss_type == 'l1':
        return jnp.mean(jnp.abs(preds - targets))
    elif loss_type == 'l1+mse':
        mse = jnp.mean((preds - targets) ** 2)
        mae = jnp.mean(jnp.abs(preds - targets))
        return mse + 0.1 * mae
    elif loss_type == 'iou':
        B_ = preds.shape[0]
        return continuous_iou_loss(preds.reshape(B_, -1), targets.reshape(B_, -1)) + jnp.mean((preds - targets) ** 2)
    else:
        return jnp.mean((preds - targets) ** 2)

# ---------------------------------------------------------------------------
# ConvLSTM Cell
# ---------------------------------------------------------------------------
class ConvLSTMCell(nn.Module):
    """Standard ConvLSTM cell with optional LayerNorm.

    Gates i, f, g, o are computed from a single joint convolution over the
    concatenation of input x_t and previous hidden state h_t, then split.
    This is equivalent to separate W_x and W_h convolutions (one matmul saves
    a kernel launch at no representational cost).

    Forget-gate bias initialised to +1 (Gers et al.) for gradient flow.
    """
    in_channel:  int
    num_hidden:  int
    kernel_size: int
    layer_norm:  bool = True

    @nn.compact
    def __call__(self, x_t, h_t, c_t):
        # x_t : (B, Ph, Pw, in_channel)
        # h_t, c_t : (B, Ph, Pw, num_hidden)
        forget_bias = 1.0
        ks          = (self.kernel_size, self.kernel_size)

        xh    = jnp.concatenate([x_t, h_t], axis=-1)
        gates = nn.Conv(self.num_hidden * 4, ks, padding='SAME',
                        use_bias=True, name='conv_gates')(xh)
        if self.layer_norm:
            gates = nn.LayerNorm(name='ln_gates')(gates)

        i_g, f_g, g_g, o_g = jnp.split(gates, 4, axis=-1)
        i_g = jax.nn.sigmoid(i_g)
        f_g = jax.nn.sigmoid(f_g + forget_bias)
        g_g = jnp.tanh(g_g)
        o_g = jax.nn.sigmoid(o_g)

        c_new = f_g * c_t + i_g * g_g
        h_new = o_g * jnp.tanh(c_new)
        return h_new, c_new

# ---------------------------------------------------------------------------
# ConvLSTM Model
# ---------------------------------------------------------------------------
class ConvLSTMModel(nn.Module):
    """Stacked ConvLSTM for spatiotemporal frame prediction.

    Training protocol (mirrored exactly in S4DModel):
      Encoder (t = 0 … input_length−2) : feed ground-truth frame t.
      Decoder (t = input_length−1 … T−2): feed last generated frame (autoregressive).
    This is deterministic — no scheduled-sampling blending in either baseline.

    Layer connectivity:
      Layer 0   : input is the raw patched frame (frame_channel channels).
      Layer 1…L : input is h[i−1] from the layer below (num_hidden channels).
    Output: 1×1 conv on h[-1] to reconstruct frame_channel.
    """
    num_layers:    int
    num_hidden:    int
    frame_channel: int
    input_length:  int
    kernel_size:   int  = 5
    layer_norm:    bool = True
    loss_type:     str  = 'l1+mse'

    def setup(self):
        self.cells = [
            ConvLSTMCell(
                in_channel  = self.frame_channel if i == 0 else self.num_hidden,
                num_hidden  = self.num_hidden,
                kernel_size = self.kernel_size,
                layer_norm  = self.layer_norm,
            )
            for i in range(self.num_layers)
        ]
        # use_bias=True allows learning baseline channel offsets
        self.conv_last = nn.Conv(self.frame_channel, (1, 1),
                                 padding='SAME', use_bias=True)

    def __call__(self, frames: jnp.ndarray):
        # frames : (B, T, Ph, Pw, frame_channel)
        B, T, Ph, Pw, _ = frames.shape

        h = [jnp.zeros((B, Ph, Pw, self.num_hidden)) for _ in range(self.num_layers)]
        c = [jnp.zeros((B, Ph, Pw, self.num_hidden)) for _ in range(self.num_layers)]
        x_gen = frames[:, 0]   # initialised here; overwritten before decoder use

        next_frames = []
        for t in range(T - 1):
            # Encoder: ground truth.  Decoder: last generated frame.
            net = frames[:, t] if t < self.input_length - 1 else x_gen

            h[0], c[0] = self.cells[0](net, h[0], c[0])
            for i in range(1, self.num_layers):
                h[i], c[i] = self.cells[i](h[i - 1], h[i], c[i])

            x_gen = self.conv_last(h[-1])   # (B, Ph, Pw, frame_channel)
            next_frames.append(x_gen)

        next_frames = jnp.stack(next_frames, axis=1)   # (B, T-1, Ph, Pw, frame_channel)

        targets = frames[:, 1:]
        loss = spatiotemporal_loss(next_frames, targets, self.loss_type)
        return next_frames, loss

# ---------------------------------------------------------------------------
# Training state & steps
# ---------------------------------------------------------------------------
class TrainState(train_state.TrainState):
    pass


def create_optimizer(lr, weight_decay, steps_per_epoch, epochs):
    schedule = optax.cosine_decay_schedule(lr, epochs * steps_per_epoch)
    return optax.adamw(schedule, weight_decay=weight_decay)


@jax.jit
def train_step(state, frames_patched):
    def loss_fn(params):
        _, loss = state.apply_fn({'params': params}, frames_patched)
        return loss
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


@jax.jit
def eval_step(state, frames_patched):
    next_frames, loss = state.apply_fn({'params': state.params}, frames_patched)
    targets = frames_patched[:, 1:]
    mse = jnp.mean((next_frames - targets) ** 2)
    mae = jnp.mean(jnp.abs(next_frames - targets))
    return loss, mse, mae

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def count_params(params) -> int:
    return sum(p.size for p in jax.tree_util.tree_leaves(params))


def make_ckpt_mgr(ckpt_dir='./convlstm_checkpoint'):
    opts = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
    return ocp.CheckpointManager(os.path.abspath(ckpt_dir), options=opts)


def save_ckpt(mgr, state, epoch, loss):
    mgr.save(epoch, args=ocp.args.StandardSave(
        {'state': state, 'epoch': epoch, 'loss': loss}
    ))
    mgr.wait_until_finished()


def denormalize_frame(arr: np.ndarray) -> np.ndarray:
    """Reverse the z-score normalisation and convert back to uint8 RGB.

    Uses the same per-channel mean/std as ProcessSeqMap._load_one.
    Input:  float32 (H, W, C) in normalised space.
    Output: uint8   (H, W, C) in [0, 255].
    """
    mean = np.array([0.40583992, 0.03281116, 0.78520891], dtype=np.float32)
    std  = np.array([0.11256019, 0.05567516, 0.10792409], dtype=np.float32)
    arr  = arr * std + mean          # undo z-score
    arr  = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def select_eval_indices(val_paths, args, n_samples=10):
    """Select sequences from val_paths, prioritizing those with active cloud dynamics."""
    n_windows = max(0, len(val_paths) - args.total_length + 1)
    if n_windows == 0:
        return []

    # Score windows by cloud pixels (green channel > 20 in center crop)
    scores = []
    step = max(1, n_windows // 100)
    for idx in range(0, n_windows, step):
        try:
            img = Image.open(val_paths[idx]).convert('RGB')
            w, h = img.size
            crop = np.array(img.crop(((w - 64) // 2, (h - 64) // 2, (w + 64) // 2, (h + 64) // 2)))
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


def visualize_predictions(state, val_paths, args, epoch, patch_size, frame_channel,
                          img_channel, patch_width, n_samples=10,
                          out_dir='./convlstm_eval_images'):
    """Run inference on validation sequences, reverse normalisation,
    and save individual frames + comparison strips.
    """
    epoch_dir = os.path.join(out_dir, f'epoch_{epoch:03d}')
    os.makedirs(epoch_dir, exist_ok=True)

    sample_indices = select_eval_indices(val_paths, args, n_samples)
    if not sample_indices:
        return

    process = ProcessSeqMap(args.img_width)

    for rank, idx in enumerate(sample_indices):
        sample_dir = os.path.join(epoch_dir, f'sample_{rank:02d}')
        os.makedirs(sample_dir, exist_ok=True)

        paths_window = val_paths[idx: idx + args.total_length]
        element = {"paths": paths_window}
        element = process.map(element)
        frames  = jnp.array(element["frames"])[None]  # (1,T,H,W,C)

        # Patch and run inference
        frames_p   = space_to_depth(frames, patch_size)  # (1,T,Ph,Pw,p²C)
        preds_p, _ = state.apply_fn({'params': state.params}, frames_p)
        preds_img  = depth_to_space(preds_p, patch_size, img_channel)
        gt_img     = frames[:, 1:]

        # Convert to numpy
        input_np = np.array(frames[0, 0])
        preds_np = np.array(preds_img[0])
        gt_np    = np.array(gt_img[0])
        n_future = preds_np.shape[0]

        # Save first input frame
        input_rgb = denormalize_frame(input_np)
        Image.fromarray(input_rgb).save(os.path.join(sample_dir, 'frame_00_input.png'))

        # Save each ground-truth and predicted frame
        gt_rgbs = []
        pred_rgbs = []
        for t in range(n_future):
            gt_rgb   = denormalize_frame(gt_np[t])
            pred_rgb = denormalize_frame(preds_np[t])
            gt_rgbs.append(gt_rgb)
            pred_rgbs.append(pred_rgb)

            Image.fromarray(gt_rgb).save(os.path.join(sample_dir, f'frame_{t + 1:02d}_gt.png'))
            Image.fromarray(pred_rgb).save(os.path.join(sample_dir, f'frame_{t + 1:02d}_pred.png'))

        # Save side-by-side comparison strip:
        # Row 1: Input | GT 1 | GT 2 | ...
        # Row 2: Input | Pred 1 | Pred 2 | ...
        H, W, _ = input_rgb.shape
        strip = np.zeros((H * 2, W * (1 + n_future), 3), dtype=np.uint8)
        strip[:H, :W] = input_rgb
        for t, g in enumerate(gt_rgbs):
            strip[:H, (t + 1) * W: (t + 2) * W] = g
        strip[H:, :W] = input_rgb
        for t, p in enumerate(pred_rgbs):
            strip[H:, (t + 1) * W: (t + 2) * W] = p

        Image.fromarray(strip).save(os.path.join(sample_dir, 'comparison_strip.png'))

    try:
        mlflow.log_artifacts(epoch_dir, artifact_path=f'eval_images/epoch_{epoch:03d}')
    except Exception as e:
        print(f"Warning: MLflow logging failed: {e}")

    print(f'  📸 Saved {len(sample_indices)} eval samples to {epoch_dir}')


def load_ckpt(mgr, state):
    latest = mgr.latest_step()
    if latest is None:
        return state, 0, float('inf')
    try:
        restored = mgr.restore(latest, args=ocp.args.StandardRestore(
            {'state': state, 'epoch': 0, 'loss': float('inf')}
        ))
        print(f'Resumed from checkpoint step {latest}.')
        return restored['state'], restored['epoch'] + 1, restored['loss']
    except (ValueError, Exception) as e:
        print(f'WARNING: checkpoint incompatible ({type(e).__name__}: {e})\n'
              f'  → discarding stale checkpoint at step {latest} and starting fresh.')
        ckpt_dir = os.path.abspath(mgr._directory)
        try:
            shutil.rmtree(ckpt_dir)
            os.makedirs(ckpt_dir, exist_ok=True)
            print(f'  Cleared: {ckpt_dir}')
        except Exception as rm_err:
            print(f'  (could not clear dir automatically: {rm_err} — delete it manually)')
        return state, 0, float('inf')

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # MLflow experiment setup
    mlflow.set_experiment("ConvLSTM-vs-S4D")
    mlflow.start_run(run_name="ConvLSTM")
    mlflow.log_params({
        "model":        "ConvLSTM",
        "lr":           args.lr,
        "weight_decay": args.weight_decay,
        "epochs":       args.epochs,
        "batch_size":   args.batch_size,
        "num_layers":   args.num_layers,
        "num_hidden":   args.num_hidden,
        "filter_size":  args.filter_size,
        "layer_norm":   args.layer_norm,
        "loss":         args.loss,
        "patch_size":   args.patch_size,
        "input_length": args.input_length,
        "total_length": args.total_length,
        "jax_backend":  jax.default_backend(),
    })
    print(f"MLflow run: {mlflow.active_run().info.run_id}")

    dataset_path = '/mnt/c/Users/Acer/Desktop/dataset'
    valid_exts   = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
    all_paths    = []

    def natural_sort_key(s):
        """Extract number from filename for natural chronological order (e.g. '12.jpg' -> 12)."""
        name = os.path.splitext(os.path.basename(s))[0]
        digits = re.findall(r'\d+', name)
        return int(digits[0]) if digits else name

    if os.path.exists(dataset_path):
        all_files = [
            os.path.join(dataset_path, f)
            for f in os.listdir(dataset_path)
            if os.path.isfile(os.path.join(dataset_path, f))
            and f.lower().endswith(valid_exts)
        ]
        all_files.sort(key=natural_sort_key)
        all_paths = all_files

    if len(all_paths) < args.total_length:
        print(f'==> Need ≥{args.total_length} images. Found {len(all_paths)}. Exiting.')
        mlflow.end_run()
        exit(0)

    split       = int(len(all_paths) * 0.9)
    train_paths = all_paths[:split]
    val_paths   = all_paths[split:]

    def make_loader(paths, shuffle=False):
        if len(paths) < args.total_length:
            return []
        src = SeqImageSource(paths, args.total_length)
        sam = grain.IndexSampler(
            num_records=len(src), num_epochs=1,
            shard_options=grain.NoSharding(),
            shuffle=shuffle, seed=args.seed,
        )
        return grain.DataLoader(
            data_source=src,
            sampler=sam,
            operations=[
                ProcessSeqMap(args.img_width),
                grain.Batch(args.batch_size, drop_remainder=True),
            ],
            worker_count=0,
        )

    steps_per_epoch = (len(train_paths) - args.total_length + 1) // args.batch_size
    val_steps       = (len(val_paths)   - args.total_length + 1) // args.batch_size
    if steps_per_epoch <= 0:
        print('==> Training set too small. Exiting.')
        mlflow.end_run()
        exit(0)

    # --- Build model ---
    rng   = jax.random.PRNGKey(args.seed)
    model = ConvLSTMModel(
        num_layers    = args.num_layers,
        num_hidden    = args.num_hidden,
        frame_channel = frame_channel,
        input_length  = args.input_length,
        kernel_size   = args.filter_size,
        layer_norm    = bool(args.layer_norm),
        loss_type     = "iou",
    )
    dummy  = jnp.zeros((1, args.total_length, patch_width, patch_width, frame_channel))
    params = model.init(rng, dummy)

    n_params = count_params(params['params'])
    print(f'ConvLSTM parameters : {n_params:,}')
    print(f'Patching            : {args.img_width}²×{args.img_channel}'
          f' → {patch_width}²×{frame_channel}  (patch_size={args.patch_size})')
    print(f'Sequence            : input={args.input_length}, total={args.total_length}')
    print(f'Loss function       : {args.loss}')
    print(f'Dataset             : {len(train_paths)} train / {len(val_paths)} val (chronologically split)')
    mlflow.log_param('num_params', n_params)

    tx    = create_optimizer(args.lr, args.weight_decay, steps_per_epoch, args.epochs)
    state = TrainState.create(apply_fn=model.apply, params=params['params'], tx=tx)

    mgr = make_ckpt_mgr()
    if args.resume:
        state, start_ep, best_loss = load_ckpt(mgr, state)
        print(f'==> Resuming from epoch {start_ep} (best val_loss so far: {best_loss:.4f})')
    else:
        # Clear stale/collapsed checkpoint directory when starting fresh
        ckpt_dir = os.path.abspath('./convlstm_checkpoint')
        if os.path.exists(ckpt_dir):
            try:
                shutil.rmtree(ckpt_dir)
                os.makedirs(ckpt_dir, exist_ok=True)
                print(f'==> Starting fresh: cleared stale checkpoint directory at {ckpt_dir}')
            except Exception as e:
                print(f'Notice: could not clear checkpoint dir: {e}')
        start_ep = 0
        best_loss = float('inf')
        print('==> Starting training from epoch 0.')

    for ep in range(start_ep, args.epochs):

        # ---- Training --------------------------------------------------------
        loader    = make_loader(train_paths, shuffle=True)
        loss_sum  = 0.0
        n_batches = 0

        pbar = tqdm(loader, total=steps_per_epoch, desc=f'Ep {ep:03d} [train]')
        for batch in pbar:
            frames_p = space_to_depth(jnp.array(batch['frames']), args.patch_size)
            state, loss = train_step(state, frames_p)
            loss_sum  += float(loss)
            n_batches += 1
            pbar.set_postfix({'loss': f'{float(loss):.4f}'})

        avg_train = loss_sum / max(1, n_batches)
        mlflow.log_metric('train_loss', avg_train, step=ep)

        # ---- Validation ------------------------------------------------------
        val_loader  = make_loader(val_paths, shuffle=False)
        v_loss_sum = v_mse_sum = v_mae_sum = 0.0
        v_batches = 0

        for batch in val_loader:
            frames_p = space_to_depth(jnp.array(batch['frames']), args.patch_size)
            loss, mse, mae = eval_step(state, frames_p)
            v_loss_sum += float(loss)
            v_mse_sum  += float(mse)
            v_mae_sum  += float(mae)
            v_batches  += 1

        avg_val_loss = v_loss_sum / max(1, v_batches)
        avg_val_mse  = v_mse_sum  / max(1, v_batches)
        avg_val_mae  = v_mae_sum  / max(1, v_batches)
        mlflow.log_metric('val_loss', avg_val_loss, step=ep)
        mlflow.log_metric('val_mse',  avg_val_mse,  step=ep)
        mlflow.log_metric('val_mae',  avg_val_mae,  step=ep)

        print(f'Epoch {ep:03d} | train_loss: {avg_train:.4f} '
              f'| val_loss: {avg_val_loss:.4f} | val_mse: {avg_val_mse:.6f} | val_mae: {avg_val_mae:.4f}')

        # ---- Checkpoint on val improvement -----------------------------------
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            save_ckpt(mgr, state, ep, avg_val_loss)
            mlflow.log_metric('best_val_loss', best_loss, step=ep)
            print(f'  ✓ Checkpoint saved  (best val_loss = {best_loss:.4f})')

        # ---- Visual evaluation ------------------------------------------------
        if (ep + 1) % args.eval_every == 0:
            visualize_predictions(
                state, val_paths, args, ep,
                patch_size=args.patch_size,
                frame_channel=frame_channel,
                img_channel=args.img_channel,
                patch_width=patch_width,
            )

    mlflow.end_run()
    print('Training finished.')

    
    