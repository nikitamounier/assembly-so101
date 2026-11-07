#!/usr/bin/env bash
set -euo pipefail
contact_plugin_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
contact_python=${PYTHON:-python}
contact_mujoco_dir=$("$contact_python" -c 'import pathlib,mujoco; assert mujoco.__version__=="4.0.0", "Contact search adapter requires MuJoCo 4.0.0"; print(pathlib.Path(mujoco.__file__).parent)')
mkdir -p "$contact_plugin_dir/build"
if [[ "$(uname -s)" == Darwin ]]; then
  contact_library=$(find "$contact_mujoco_dir" -name 'libmujoco*.dylib' -print -quit)
  contact_link=(-dynamiclib)
else
  contact_library=$(find "$contact_mujoco_dir" -maxdepth 1 -name 'libmujoco.so*' -print -quit)
  contact_link=(-shared -fPIC)
fi
[[ -n "$contact_library" ]] || { echo "libmujoco not found in $contact_mujoco_dir" >&2; exit 1; }
contact_temp=$(mktemp "$contact_plugin_dir/build/.contact-search.XXXXXX.so")
trap 'rm -f -- "$contact_temp"' EXIT
${CC:-cc} "${contact_link[@]}" -O2 -Wall -Werror -I"$contact_mujoco_dir/include" \
  "$contact_plugin_dir/so101_contact_search.c" "$contact_library" \
  -Wl,-rpath,"$contact_mujoco_dir" -o "$contact_temp"
mv -f -- "$contact_temp" "$contact_plugin_dir/build/libso101_contact_search.so"
printf '%s\n' "$contact_plugin_dir/build/libso101_contact_search.so"
