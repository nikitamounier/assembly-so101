#!/usr/bin/env bash
# Build the AssembleNIST SDF plugin (bolt/nut thread SDFs and connector grid SDFs) against the headers and
# library shipped with the `mujoco` Python package (x86 F16C or native ARM64 half conversion).
# Output: plugin/build/libassemblenist_sdf.so
#   bash plugin/build.sh                 # uses `python` on PATH
#   PYTHON=/path/to/python bash plugin/build.sh
#   MUJOCO_DIR=/path/to/site-packages/mujoco bash plugin/build.sh
set -euo pipefail
cd "$(dirname "$0")"
MJ=${MUJOCO_DIR:-$("${PYTHON:-python}" -c "import mujoco, os; print(os.path.dirname(mujoco.__file__))")}
if [[ "$(uname -s)" == Darwin ]]; then
  LIB=$(find "$MJ" -name 'libmujoco*.dylib' -print -quit)
  TASK_LINK=(-dynamiclib -Wl,-install_name,@rpath/libassemblenist_sdf.so)
else
  LIB=$(find "$MJ" -maxdepth 1 -name 'libmujoco.so*' -print -quit)
  TASK_LINK=(-shared -fPIC -Wl,-soname,libassemblenist_sdf.so)
fi
TASK_ARCH=(-fPIC)
case "$(uname -m)" in x86_64|amd64) TASK_ARCH+=(-mf16c);; esac
[ -n "$LIB" ] || { echo "libmujoco not found in $MJ"; exit 1; }
mkdir -p build
TASK_PLUGIN_TEMP=$(mktemp build/.assemblenist.XXXXXX.so)
trap 'rm -f -- "$TASK_PLUGIN_TEMP"' EXIT
# Keep the established .so filename on both hosts; macOS dlopen accepts Mach-O libraries with this suffix.
${CXX:-g++} -O3 -std=c++17 "${TASK_ARCH[@]}" "${TASK_LINK[@]}" -I"$MJ/include" assemblenist_sdf.cc -o "$TASK_PLUGIN_TEMP" "$LIB" \
  -Wl,-rpath,"$MJ"
mv -f -- "$TASK_PLUGIN_TEMP" build/libassemblenist_sdf.so
echo "built plugin/build/libassemblenist_sdf.so against $LIB"
