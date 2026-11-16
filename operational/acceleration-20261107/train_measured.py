"""Benchmark wrapper: unchanged Trainer with per-iteration wall/checkpoint timing."""

import json
import signal
import time
from pathlib import Path

import torch

from assembly_policy.train import Trainer, parse

args = parse()
trainer = Trainer(args)
trainer.stop_requested = False
signal.signal(signal.SIGTERM, lambda *_: setattr(trainer, "stop_requested", True))
metadata = dict(
    gpu=torch.cuda.get_device_name(),
    gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
    torch=torch.__version__,
    source_root=str(Path.cwd()),
    args=trainer.args,
)
(Path(args.run_dir) / "hardware.json").write_text(json.dumps(metadata, indent=2) + "\n")
periods = []
collect, update, save = trainer.collect, trainer.update, trainer.save


def measured_collect():
    now = time.perf_counter()
    if periods:
        periods[-1]["iteration_s"] = now - periods[-1]["started"]
    periods.append(dict(iteration=trainer.it + 1, started=now, collect_s=0, update_s=0, save_s=0))
    result = collect()
    periods[-1]["collect_s"] = time.perf_counter() - now
    return result


def measured_update(batch):
    start = time.perf_counter()
    result = update(batch)
    periods[-1]["update_s"] += time.perf_counter() - start
    return result


def measured_save(*args, **kwargs):
    start = time.perf_counter()
    result = save(*args, **kwargs)
    if periods:
        periods[-1]["save_s"] += time.perf_counter() - start
    return result


trainer.collect = measured_collect
trainer.update = measured_update
trainer.save = measured_save
try:
    trainer.train()
    if periods:
        periods[-1]["iteration_s"] = time.perf_counter() - periods[-1]["started"]
    (Path(args.run_dir) / "iteration-times.json").write_text(json.dumps(periods, indent=2) + "\n")
finally:
    trainer.env.close()
for child in trainer._children:
    child.wait()
