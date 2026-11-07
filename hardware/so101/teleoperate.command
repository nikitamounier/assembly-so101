#!/usr/bin/env bash
cd -- "$(dirname -- "$0")" || exit 1

if [[ ! -x .venv/bin/python ]]; then
    printf '%s\n' 'Set up the hardware Python environment as described in README.md.'
    exit 1
fi

if (( $# == 0 )); then
    printf 'SO-101 teleoperation\n  1. Both pairs\n  2. Front WowRobo pair\n  3. Rear mixed-vendor pair\n'
    printf 'Choose 1, 2, or 3: '
    read -r selection || exit 0
    case "$selection" in
        1) set -- --both ;;
        2) set -- --front-wowrobo ;;
        3) set -- --rear ;;
        *) exit 1 ;;
    esac
fi

exec .venv/bin/python -u teleoperate.py "$@"
