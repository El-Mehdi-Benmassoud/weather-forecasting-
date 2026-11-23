import os, sys
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

ckpt_path = os.path.abspath('./checkpoint')
mngr = ocp.CheckpointManager(ckpt_path)
step = mngr.latest_step()
print(f"Latest step: {step}")

ckpt_dir = os.path.join(ckpt_path, str(step), 'default')

# Try PyTreeCheckpointer to load raw checkpoint without target
ckptr = ocp.PyTreeCheckpointer()
try:
    raw = ckptr.restore(ckpt_dir)
    print("SUCCESS with PyTreeCheckpointer")
    if isinstance(raw, dict):
        for k in raw:
            print(f"  key: {k}, type: {type(raw[k])}")
            if k == 'state' and hasattr(raw[k], 'params'):
                params = raw[k].params
                print(f"  params keys: {list(params.keys()) if isinstance(params, dict) else type(params)}")
except Exception as e:
    print(f"PyTreeCheckpointer failed: {type(e).__name__}: {e}")

# Try StandardCheckpointer
try:
    sc = ocp.StandardCheckpointer()
    raw2 = sc.restore(ckpt_dir)
    print("SUCCESS with StandardCheckpointer")
except Exception as e:
    print(f"StandardCheckpointer failed: {type(e).__name__}: {str(e)[:200]}")
