"""Run an isolated gravity/contact final-module demonstration."""

import argparse
import json

from assembly_policy.so101.wowrobo_passive_final import run

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--seconds", type=float, default=12)
p.add_argument("--no-own-contact", action="store_true")
p.add_argument("--kp", type=float, default=1000)
p.add_argument("--kd", type=float, default=25)
p.add_argument("--preload", type=float, default=0)
a = p.parse_args()
print(
    json.dumps(
        run(
            a.out,
            seconds=a.seconds,
            own_contact=not a.no_own_contact,
            kp=a.kp,
            kd=a.kd,
            preload=a.preload,
        ),
        indent=2,
    )
)
