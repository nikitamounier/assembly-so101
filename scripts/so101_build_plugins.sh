#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
TASK_PYTHON=${PYTHON:-"$PWD/.venv/bin/python"}
PYTHON="$TASK_PYTHON" bash plugin/build.sh
TASK_MJ=$("$TASK_PYTHON" -c 'import mujoco, pathlib; print(pathlib.Path(mujoco.__file__).parent)')
if [[ "$(uname -s)" == Darwin ]]; then
    TASK_LIB=$(find "$TASK_MJ" -name 'libmujoco*.dylib' -print -quit)
    TASK_LINK=(-dynamiclib)
else
    TASK_LIB=$(find "$TASK_MJ" -maxdepth 1 -name 'libmujoco.so*' -print -quit)
    TASK_LINK=(-shared -fPIC)
fi
[[ -n "$TASK_LIB" ]] || { echo "libmujoco not found in $TASK_MJ" >&2; exit 1; }
TASK_DRIVER_TEMP=$(mktemp plugin/build/.driver.XXXXXX.so)
trap 'rm -f -- "$TASK_DRIVER_TEMP"' EXIT
${CXX:-g++} -O3 -std=c++17 "${TASK_LINK[@]}" -I"$TASK_MJ/include" plugin/so101_driver.cc \
    -o "$TASK_DRIVER_TEMP" "$TASK_LIB" -Wl,-rpath,"$TASK_MJ"
mv -f -- "$TASK_DRIVER_TEMP" plugin/build/libso101_driver.so
PYTHON="$TASK_PYTHON" bash plugin/build_contact_search.sh
