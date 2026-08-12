#pragma once

#include <cstdint>

namespace pdl_gemm_ar {
namespace fp4 {

// The experiment uses the positive E2M1 magnitudes 0, .5, 1, 1.5, 2, 3, 4,
// and 6.  Hopper has no multimem FP4 reduction instruction, so encoding and
// decoding are deliberately kept as small CUDA-core helpers.
__device__ __forceinline__ uint8_t encode_nibble(float value) {
    if (!isfinite(value) || value == 0.0f) return 0;
    const bool negative = value < 0.0f;
    const float magnitude = fabsf(value);
    uint8_t code;
    if (magnitude < 0.25f) code = 0;
    else if (magnitude < 0.75f) code = 1;
    else if (magnitude < 1.25f) code = 2;
    else if (magnitude < 1.75f) code = 3;
    else if (magnitude < 2.5f) code = 4;
    else if (magnitude < 3.5f) code = 5;
    else if (magnitude < 5.0f) code = 6;
    else code = 7;
    return static_cast<uint8_t>(code | (negative ? 0x8 : 0));
}

__device__ __forceinline__ float decode_nibble(uint8_t code) {
    constexpr float magnitudes[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
    };
    float value = magnitudes[code & 0x7];
    return (code & 0x8) ? -value : value;
}

__device__ __forceinline__ uint8_t encode_pair(float first, float second) {
    return static_cast<uint8_t>(
        encode_nibble(first) | (encode_nibble(second) << 4)
    );
}

__device__ __forceinline__ float rounded_bf16(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

__device__ __forceinline__ __nv_bfloat16 decode_scale(float amax) {
    return __float2bfloat16_rn(amax / 6.0f);
}

__device__ __forceinline__ float scale_as_float(__nv_bfloat16 scale) {
    return __bfloat162float(scale);
}

__device__ __forceinline__ float normalize(float value, float scale) {
    // The E2M1 decision boundaries are discrete.  --use_fast_math turns a
    // reciprocal followed by multiply into an approximation that can move an
    // exact BF16 ratio across 0.25/0.75/... and flip a nibble.  Force RN
    // division so the register path matches the wire reference bit-for-bit.
    return scale == 0.0f ? 0.0f : __fdiv_rn(value, scale);
}

__device__ __forceinline__ uint16_t bf16_bits(__nv_bfloat16 value) {
    return *reinterpret_cast<const uint16_t *>(&value);
}

__device__ __forceinline__ uint32_t pack_bf16x2_bits(
    __nv_bfloat16 first,
    __nv_bfloat16 second
) {
    return static_cast<uint32_t>(bf16_bits(first))
        | (static_cast<uint32_t>(bf16_bits(second)) << 16);
}

template <int WIDTH>
__device__ __forceinline__ float subgroup_max(float value) {
    static_assert(WIDTH == 4 || WIDTH == 8);
    unsigned mask = 0xffffffffu;
    #pragma unroll
    for (int offset = WIDTH / 2; offset > 0; offset >>= 1)
        value = fmaxf(value, __shfl_xor_sync(mask, value, offset, WIDTH));
    return value;
}

__device__ __forceinline__ void multicast_store_u32(uint32_t *ptr, uint32_t value) {
    // Use a system-scope release store: the tile-ready red.release.sys signal
    // is issued by a different warp after a CTA barrier, so weak stores here
    // are not sufficient to publish the payload to peer GPUs before their
    // acquire wait completes.  bf16x2 preserves the 32 payload bits exactly.
    asm volatile(
        "multimem.st.release.sys.global.bf16x2 [%0], %1;"
        :: "l"(ptr), "r"(value) : "memory"
    );
}

__device__ __forceinline__ void peer_store_u32(uint32_t *ptr, uint32_t value) {
    asm volatile("st.global.u32 [%0], %1;" :: "l"(ptr), "r"(value) : "memory");
}

__device__ __forceinline__ uint32_t gather_u8x4(uint8_t value, int lane) {
    const int base = lane & ~3;
    uint32_t packed = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        packed |= static_cast<uint32_t>(
            __shfl_sync(0xffffffffu, static_cast<unsigned>(value), base + i)
        ) << (8 * i);
    }
    return packed;
}

} // namespace fp4
} // namespace pdl_gemm_ar
