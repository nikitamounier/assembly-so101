"""Preserve upstream SDF equations while accepting the exact platform shim."""

PORTABLE_HALF = """inline float h2f(uint16_t x) {
#if defined(__F16C__)
  return _cvtsh_ss(x);
#elif defined(__aarch64__)
  __fp16 value;
  std::memcpy(&value, &x, sizeof(x));
  return static_cast<float>(value);
#else
#error "SDF grids require x86 F16C or ARM64 half-precision conversion"
#endif
}"""


def sdf_equations_unchanged(original, current):
    # Normalize only these two exact portability additions. A changed conversion,
    # thread equation or grid evaluator must still fail the original source gate.
    current = current.replace(
        "#if defined(__F16C__)\n#include <immintrin.h>\n#endif", "#include <immintrin.h>"
    )
    current = current.replace(PORTABLE_HALF, "inline float h2f(uint16_t x) { return _cvtsh_ss(x); }")
    # The acceleration helper is separate; the original SDF equations and
    # callbacks must still match exactly after removing this one include.
    current = current.removesuffix('\n#include "so101_grid_bounds.h"\n')
    return (
        original.split("const Grid* get_grid")[0] == current.split("const Grid* get_grid")[0]
        and original.split("inline float h2f")[1] == current.split("inline float h2f")[1]
    )
