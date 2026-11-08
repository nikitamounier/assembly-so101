"""Launch an isolated, explicitly identified Warp run of the current trainer.

Usage: python scripts/train_wowrobo_warp.py --trainer-root /path/to/assembly-so101
  --scratch /path/to/private/scratch -- <normal assembly_policy.train arguments>
Requires the current reset/warmstart/friction-aware WowRobo trainer. Physics
code stays in this canonical repository; the trainer is supplied explicitly.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer-root", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--jacobian", choices=("native", "dense"), default="dense")
    parser.add_argument("--nconmax", type=int, default=512)
    parser.add_argument("--njmax", type=int, default=2048)
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(options.trainer_root.resolve()))
    os.environ.setdefault(
        "WOWROBO_WARP_GRID_SOURCE",
        str(options.trainer_root / "plugin/assemblenist_sdf.cc"),
    )
    os.environ.setdefault(
        "WOWROBO_WARP_FRICTION_SOURCE",
        str(root / "assembly_policy/so101/wowrobo_friction.py"),
    )
    from assembly_policy import train
    from assembly_policy.wowrobo import env as backend

    for name in ("wowrobo_warp", "wowrobo_warp_runtime"):
        spec = importlib.util.spec_from_file_location(
            "assembly_policy.so101." + name,
            root / "assembly_policy/so101" / (name + ".py"),
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    runtime = sys.modules["assembly_policy.so101.wowrobo_warp_runtime"]
    args = train.parse(
        options.trainer_args[1:]
        if options.trainer_args[:1] == ["--"]
        else options.trainer_args
    )
    if args.suite != "wowrobo" or args.remote_workers:
        parser.error("Warp runs require --suite wowrobo --remote-workers 0")
    if not args.no_eval:
        parser.error("Use --no-eval until the evaluator supports this distinct backend")
    engine_options = {
        "jacobian": options.jacobian,
        "nconmax": options.nconmax,
        "njmax": options.njmax,
    }
    native_class = backend.BatchedAssemblyEnv

    class WarpEnvironment(native_class):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            try:
                runtime.install_on_environment(self, options.scratch, **engine_options)
            except BaseException:
                self.close()
                raise

    backend.BatchedAssemblyEnv = WarpEnvironment
    native_trainer = train.Trainer

    class WarpTrainer(native_trainer):
        def _maybe_resume(self):
            # Executed before checkpoint comparison and before config is written.
            # Existing native run directories therefore fail the trainer's guard.
            identity = {
                "backend": "wowrobo-mjwarp-runtime-v1",
                "options": engine_options,
                "versions": {
                    n: runtime.version(n)
                    for n in ("mujoco", "mujoco-warp", "warp-lang")
                },
                "native_model_signature": self.args["model_signature"],
                "native_sources": {
                    key: hashlib.sha256(Path(os.environ[key]).read_bytes()).hexdigest()
                    for key in (
                        "WOWROBO_WARP_GRID_SOURCE",
                        "WOWROBO_WARP_FRICTION_SOURCE",
                    )
                },
                "source": {
                    n: hashlib.sha256(
                        (root / "assembly_policy/so101" / n).read_bytes()
                    ).hexdigest()
                    for n in ("wowrobo_warp.py", "wowrobo_warp_runtime.py")
                },
            }
            self.args["physics_backend"] = identity["backend"]
            self.args["physics_signature"] = hashlib.sha256(
                json.dumps(identity, sort_keys=True).encode()
            ).hexdigest()
            self.args["warp_runtime"] = identity
            self.args["run_status"] = "EXPERIMENTAL_WARP_UNQUALIFIED"
            return super()._maybe_resume()

    train.Trainer = WarpTrainer
    # Reuse the trainer's signal handling, sampler, PPO, checkpointing and close.
    train.parse = lambda: args
    train.main()


if __name__ == "__main__":
    main()
