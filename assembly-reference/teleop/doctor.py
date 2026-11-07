"""Check CPU-only scene loading, native physics and offscreen rendering."""
import json
import sys
import mujoco
import numpy as np
from .backend import load_scene,AssemblyFrames
from .runtime import Capture


def main():
    model,description=load_scene();data=mujoco.MjData(model)
    data.qpos[:]=description['home_qpos'];data.ctrl[:]=description['home_ctrl']
    mujoco.mj_step(model,data,nstep=40)
    if data.warning.number.any() or not np.isfinite(data.qpos).all():raise RuntimeError('Native physics check failed')
    frames=AssemblyFrames(model,description['cameras'],width=160,height=90)
    try:
        images,_=frames.render(Capture(tuple(data.qpos),1,b'{}'))
        from OpenGL.GL import GL_RENDERER,glGetString
        graphics=glGetString(GL_RENDERER).decode()
    finally:frames.close()
    assert set(images)=={'main','left','right'} and all(v.startswith(b'\xff\xd8') for v in images.values())
    print(json.dumps(dict(passed=True,environment=description['environment_identity'],cameras=list(images),graphics_renderer=graphics,torch_loaded='torch' in sys.modules)),flush=True)


if __name__=='__main__':main()
