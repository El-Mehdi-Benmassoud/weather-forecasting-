"""
One-time migration: Extract params from old orbax checkpoint and save as numpy .npz file.
Run this ONCE, then main.py will load from the .npz file.
"""
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.10'

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.training import train_state
import flax.traverse_util as traverse_util
import optax
import orbax.checkpoint as ocp
import math

# --- Recreate OLD model (without decoder_hidden) ---
class S4DKernel(nn.Module):
    d_model: int
    N: int = 64
    dt_min: float = 0.001
    dt_max: float = 0.1
    def setup(self):
        H, N = self.d_model, self.N
        self.log_dt = self.param('log_dt', lambda rng: jax.random.uniform(rng, (H,)) * (math.log(self.dt_max) - math.log(self.dt_min)) + math.log(self.dt_min))
        self.C_real = self.param('C_real', lambda rng: jax.random.normal(rng, (H, N // 2)))
        self.C_imag = self.param('C_imag', lambda rng: jax.random.normal(rng, (H, N // 2)))
        self.log_A_real = self.param('log_A_real', lambda rng: jnp.log(0.5 * jnp.ones((H, N // 2))))
        self.A_imag = self.param('A_imag', lambda rng: math.pi * jnp.tile(jnp.arange(N // 2), (H, 1)))
    def __call__(self, L):
        dt = jnp.exp(self.log_dt)
        C = self.C_real + 1j * self.C_imag
        A = -jnp.exp(self.log_A_real) + 1j * self.A_imag
        dtA = A * jnp.expand_dims(dt, -1)
        K = jnp.expand_dims(dtA, -1) * jnp.arange(L)
        C = C * (jnp.exp(dtA) - 1.) / A
        return 2 * jnp.einsum('hn, hnl -> hl', C, jnp.exp(K)).real

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
        if not self.transposed: u = jnp.transpose(u, (0, 2, 1))
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
        if self.transposed: y_out = jnp.transpose(y_out, (0, 2, 1))
        return y_out, None

class OldS4Model(nn.Module):
    """Old model WITHOUT decoder_hidden."""
    d_input: int = 12
    d_output: int = 4 * 224 * 224 * 3
    d_model: int = 92
    n_layers: int = 1
    dropout: float = 0.8
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
            z, _ = S4D(d_model=self.d_model, dropout=self.dropout, transposed=True, name=f's4_{i}')(z, deterministic=deterministic)
            z = nn.Dropout(rate=self.dropout, broadcast_dims=(2,), name=f'drop_{i}')(z, deterministic=deterministic)
            x = z + x
            if not self.prenorm:
                x = jnp.transpose(x, (0, 2, 1))
                x = nn.LayerNorm(name=f'norm_{i}')(x)
                x = jnp.transpose(x, (0, 2, 1))
        x = jnp.transpose(x, (0, 2, 1))
        x = jnp.mean(x, axis=1)
        x = nn.Dense(self.d_output, name='decoder')(x)
        return x

class TrainState(train_state.TrainState):
    pass

# Create old-style model and optimizer
print("Creating old model...")
old_model = OldS4Model()
rng = jax.random.PRNGKey(42)
params_key, _ = jax.random.split(rng)
dummy_input = jnp.ones((1, 224 * 224, 12))
variables = old_model.init(params_key, dummy_input, deterministic=True)
old_params = variables['params']

# Old optimizer with regular/s4 labels
scheduler = optax.cosine_decay_schedule(init_value=0.01, decay_steps=100 * 856)
s4_scheduler = optax.cosine_decay_schedule(init_value=0.001, decay_steps=100 * 856)
adamw = optax.adamw(learning_rate=scheduler, weight_decay=0.01)
s4_adam = optax.adam(learning_rate=s4_scheduler)
flat_params = traverse_util.flatten_dict(old_params)
labels = {path: 's4' if 'kernel' in path else 'regular' for path in flat_params}
old_tx = optax.multi_transform({'regular': adamw, 's4': s4_adam}, traverse_util.unflatten_dict(labels))

old_state = TrainState.create(apply_fn=old_model.apply, params=old_params, tx=old_tx)

# Restore checkpoint
print("Loading checkpoint...")
ckpt_path = os.path.abspath('./checkpoint')
options = ocp.CheckpointManagerOptions(max_to_keep=3, create=False)
mngr = ocp.CheckpointManager(ckpt_path, options=options)
step = mngr.latest_step()
print(f"Latest step: {step}")

target_dict = {'state': old_state, 'epoch': 0, 'loss': float('inf')}
restored = mngr.restore(step, args=ocp.args.StandardRestore(target_dict))

# Extract params and save as .npz
print("Extracting params...")
restored_params = restored['state'].params
flat_params = traverse_util.flatten_dict(restored_params)

# Convert to numpy arrays for saving
np_dict = {}
for key, val in flat_params.items():
    param_name = '/'.join(key)
    np_dict[param_name] = np.array(val)
    print(f"  {param_name}: {val.shape}")

epoch = int(restored['epoch'])
loss = float(restored['loss'])
np_dict['__epoch__'] = np.array(epoch)
np_dict['__loss__'] = np.array(loss)

save_path = './pretrained_params.npz'
np.savez(save_path, **np_dict)
print(f"\nSaved pretrained params to {save_path}")
print(f"  Epoch: {epoch}, Loss: {loss}")
print("Done! You can now run main.py")
