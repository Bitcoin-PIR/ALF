//! Platform-abstracted SIMD wrappers.
//!
//! Provides a V128 type alias and uniform function API that compiles
//! on both aarch64 (NEON + AES crypto) and x86_64 (SSE + AES-NI).
//! Also provides V512 wide operations for x86_64 AVX-512 VAES batching.

// ================================================================
// Platform type aliases
// ================================================================

#[cfg(target_arch = "aarch64")]
pub use std::arch::aarch64::uint8x16_t as V128;

#[cfg(target_arch = "x86_64")]
pub use std::arch::x86_64::__m128i as V128;

// ================================================================
// aarch64 NEON + AES crypto implementations
// ================================================================

#[cfg(target_arch = "aarch64")]
mod imp {
    use super::V128;
    use std::arch::aarch64::*;

    #[inline(always)]
    pub unsafe fn zero() -> V128 { vdupq_n_u8(0) }

    #[inline(always)]
    pub unsafe fn c_ff() -> V128 { vdupq_n_u8(0xFF) }

    #[inline(always)]
    pub unsafe fn xor2(a: V128, b: V128) -> V128 { veorq_u8(a, b) }

    #[inline(always)]
    pub unsafe fn xor3(a: V128, b: V128, c: V128) -> V128 { veor3q_u8(a, b, c) }

    #[inline(always)]
    pub unsafe fn and128(a: V128, b: V128) -> V128 { vandq_u8(a, b) }

    #[inline(always)]
    pub unsafe fn or128(a: V128, b: V128) -> V128 { vorrq_u8(a, b) }

    #[inline(always)]
    pub unsafe fn andnot(a: V128, b: V128) -> V128 { vbicq_u8(b, a) }

    #[inline(always)]
    pub unsafe fn shuffle(x: V128, sh: V128) -> V128 { vqtbl1q_u8(x, sh) }

    #[inline(always)]
    pub unsafe fn aesenc(state: V128, rk: V128) -> V128 {
        veorq_u8(vaesmcq_u8(vaeseq_u8(state, zero())), rk)
    }

    #[inline(always)]
    pub unsafe fn aesdec(state: V128, rk: V128) -> V128 {
        veorq_u8(vaesimcq_u8(vaesdq_u8(state, zero())), rk)
    }

    #[inline(always)]
    pub unsafe fn aesenclast(state: V128, rk: V128) -> V128 {
        veorq_u8(vaeseq_u8(state, zero()), rk)
    }

    #[inline(always)]
    pub unsafe fn aesdeclast(state: V128, rk: V128) -> V128 {
        veorq_u8(vaesdq_u8(state, zero()), rk)
    }

    #[inline(always)]
    pub unsafe fn aesimc(state: V128) -> V128 { vaesimcq_u8(state) }

    #[inline(always)]
    pub unsafe fn load128(ptr: *const u8) -> V128 { vld1q_u8(ptr) }

    #[inline(always)]
    pub unsafe fn store128(ptr: *mut u8, x: V128) { vst1q_u8(ptr, x) }

    #[inline(always)]
    pub unsafe fn combine(s1: V128, s2: V128) -> V128 {
        vorrq_u8(vqtbl1q_u8(s1, s2), vceqq_u8(s2, c_ff()))
    }

    #[inline(always)]
    pub unsafe fn blend(a: V128, b: V128, mask: V128) -> V128 { vbslq_u8(mask, b, a) }

    #[inline(always)]
    pub unsafe fn cmpeq8(a: V128, b: V128) -> V128 { vceqq_u8(a, b) }

    #[inline(always)]
    pub unsafe fn add8(a: V128, b: V128) -> V128 { vaddq_u8(a, b) }

    #[inline(always)]
    pub unsafe fn from_bytes(bytes: &[u8; 16]) -> V128 { vld1q_u8(bytes.as_ptr()) }

    #[inline(always)]
    pub unsafe fn splat8(v: u8) -> V128 { vdupq_n_u8(v) }

    #[inline(always)]
    pub unsafe fn extract_lo64(x: V128) -> u64 {
        vgetq_lane_u64::<0>(vreinterpretq_u64_u8(x))
    }

    #[inline(always)]
    pub unsafe fn extract_hi64(x: V128) -> u64 {
        vgetq_lane_u64::<1>(vreinterpretq_u64_u8(x))
    }

    #[inline(always)]
    pub unsafe fn extract_byte<const LANE: i32>(x: V128) -> u8 {
        vgetq_lane_u8::<LANE>(x)
    }

    #[inline(always)]
    pub unsafe fn from_u64x2(lo: u64, hi: u64) -> V128 {
        vreinterpretq_u8_u64(vcombine_u64(vcreate_u64(lo), vcreate_u64(hi)))
    }

    #[inline(always)]
    pub unsafe fn from_u32_low(v: u32) -> V128 {
        let arr: [u32; 4] = [v, 0, 0, 0];
        vreinterpretq_u8_u32(vld1q_u32(arr.as_ptr()))
    }

    #[inline(always)]
    pub unsafe fn bslli_3(x: V128) -> V128 { vextq_u8::<13>(zero(), x) }

    #[inline(always)]
    pub unsafe fn slli_epi64_32(x: V128) -> V128 {
        vreinterpretq_u8_u64(vshlq_n_u64::<32>(vreinterpretq_u64_u8(x)))
    }

    #[inline(always)]
    pub unsafe fn clmul_lo(a: V128, b: u64) -> V128 {
        let a_lo = vgetq_lane_u64::<0>(vreinterpretq_u64_u8(a));
        std::mem::transmute(vmull_p64(a_lo, b))
    }
}

// ================================================================
// x86_64 SSE + AES-NI implementations
// ================================================================

#[cfg(target_arch = "x86_64")]
mod imp {
    use super::V128;
    use std::arch::x86_64::*;

    #[inline(always)]
    pub unsafe fn zero() -> V128 { _mm_setzero_si128() }

    #[inline(always)]
    pub unsafe fn c_ff() -> V128 { _mm_set1_epi8(-1) }

    #[inline(always)]
    pub unsafe fn xor2(a: V128, b: V128) -> V128 { _mm_xor_si128(a, b) }

    #[inline(always)]
    pub unsafe fn xor3(a: V128, b: V128, c: V128) -> V128 {
        _mm_xor_si128(_mm_xor_si128(a, b), c)
    }

    #[inline(always)]
    pub unsafe fn and128(a: V128, b: V128) -> V128 { _mm_and_si128(a, b) }

    #[inline(always)]
    pub unsafe fn or128(a: V128, b: V128) -> V128 { _mm_or_si128(a, b) }

    /// (~a) & b  — matches x86 _mm_andnot_si128(a, b)
    #[inline(always)]
    pub unsafe fn andnot(a: V128, b: V128) -> V128 { _mm_andnot_si128(a, b) }

    #[inline(always)]
    pub unsafe fn shuffle(x: V128, sh: V128) -> V128 { _mm_shuffle_epi8(x, sh) }

    #[inline(always)]
    pub unsafe fn aesenc(state: V128, rk: V128) -> V128 { _mm_aesenc_si128(state, rk) }

    #[inline(always)]
    pub unsafe fn aesdec(state: V128, rk: V128) -> V128 { _mm_aesdec_si128(state, rk) }

    #[inline(always)]
    pub unsafe fn aesenclast(state: V128, rk: V128) -> V128 { _mm_aesenclast_si128(state, rk) }

    #[inline(always)]
    pub unsafe fn aesdeclast(state: V128, rk: V128) -> V128 { _mm_aesdeclast_si128(state, rk) }

    #[inline(always)]
    pub unsafe fn aesimc(state: V128) -> V128 { _mm_aesimc_si128(state) }

    #[inline(always)]
    pub unsafe fn load128(ptr: *const u8) -> V128 { _mm_loadu_si128(ptr as *const V128) }

    #[inline(always)]
    pub unsafe fn store128(ptr: *mut u8, x: V128) { _mm_storeu_si128(ptr as *mut V128, x) }

    #[inline(always)]
    pub unsafe fn combine(s1: V128, s2: V128) -> V128 {
        _mm_or_si128(_mm_shuffle_epi8(s1, s2), _mm_cmpeq_epi8(s2, c_ff()))
    }

    #[inline(always)]
    pub unsafe fn blend(a: V128, b: V128, mask: V128) -> V128 { _mm_blendv_epi8(a, b, mask) }

    #[inline(always)]
    pub unsafe fn cmpeq8(a: V128, b: V128) -> V128 { _mm_cmpeq_epi8(a, b) }

    #[inline(always)]
    pub unsafe fn add8(a: V128, b: V128) -> V128 { _mm_add_epi8(a, b) }

    #[inline(always)]
    pub unsafe fn from_bytes(bytes: &[u8; 16]) -> V128 {
        _mm_loadu_si128(bytes.as_ptr() as *const V128)
    }

    #[inline(always)]
    pub unsafe fn splat8(v: u8) -> V128 { _mm_set1_epi8(v as i8) }

    #[inline(always)]
    pub unsafe fn extract_lo64(x: V128) -> u64 { _mm_cvtsi128_si64(x) as u64 }

    #[inline(always)]
    pub unsafe fn extract_hi64(x: V128) -> u64 { _mm_extract_epi64::<1>(x) as u64 }

    #[inline(always)]
    pub unsafe fn extract_byte<const LANE: i32>(x: V128) -> u8 {
        _mm_extract_epi8::<LANE>(x) as u8
    }

    #[inline(always)]
    pub unsafe fn from_u64x2(lo: u64, hi: u64) -> V128 {
        _mm_set_epi64x(hi as i64, lo as i64)
    }

    #[inline(always)]
    pub unsafe fn from_u32_low(v: u32) -> V128 { _mm_cvtsi32_si128(v as i32) }

    #[inline(always)]
    pub unsafe fn bslli_3(x: V128) -> V128 { _mm_bslli_si128::<3>(x) }

    #[inline(always)]
    pub unsafe fn slli_epi64_32(x: V128) -> V128 { _mm_slli_epi64::<32>(x) }

    #[inline(always)]
    pub unsafe fn clmul_lo(a: V128, b: u64) -> V128 {
        _mm_clmulepi64_si128(a, _mm_set_epi64x(0, b as i64), 0x00)
    }
}

// Re-export all functions from the platform-specific module
pub use imp::*;

// ================================================================
// x86_64 AVX-512 VAES wide operations (4 × 128-bit lanes)
// ================================================================

#[cfg(target_arch = "x86_64")]
pub mod wide {
    use std::arch::x86_64::*;
    use super::V128;

    pub type V512 = __m512i;

    #[inline(always)]
    pub unsafe fn zero_x4() -> V512 { _mm512_setzero_si512() }

    #[inline(always)]
    pub unsafe fn broadcast(x: V128) -> V512 { _mm512_broadcast_i32x4(x) }

    /// Load 4 independent 128-bit blocks into a 512-bit register.
    #[inline(always)]
    pub unsafe fn load_4x128(b0: *const u8, b1: *const u8, b2: *const u8, b3: *const u8) -> V512 {
        let v0 = _mm_loadu_si128(b0 as *const __m128i);
        let v1 = _mm_loadu_si128(b1 as *const __m128i);
        let v2 = _mm_loadu_si128(b2 as *const __m128i);
        let v3 = _mm_loadu_si128(b3 as *const __m128i);
        let lo = _mm256_inserti128_si256(_mm256_castsi128_si256(v0), v1, 1);
        let hi = _mm256_inserti128_si256(_mm256_castsi128_si256(v2), v3, 1);
        _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1)
    }

    /// Store 4 lanes of a 512-bit register to separate buffers.
    #[inline(always)]
    pub unsafe fn store_4x128(x: V512, b0: *mut u8, b1: *mut u8, b2: *mut u8, b3: *mut u8) {
        let lo = _mm512_castsi512_si256(x);
        let hi = _mm512_extracti64x4_epi64(x, 1);
        _mm_storeu_si128(b0 as *mut __m128i, _mm256_castsi256_si128(lo));
        _mm_storeu_si128(b1 as *mut __m128i, _mm256_extracti128_si256(lo, 1));
        _mm_storeu_si128(b2 as *mut __m128i, _mm256_castsi256_si128(hi));
        _mm_storeu_si128(b3 as *mut __m128i, _mm256_extracti128_si256(hi, 1));
    }

    /// Extract one 128-bit lane from a 512-bit register.
    #[inline(always)]
    pub unsafe fn extract_lane(x: V512, lane: usize) -> V128 {
        match lane {
            0 => _mm512_castsi512_si128(x),
            1 => _mm256_extracti128_si256(_mm512_castsi512_si256(x), 1),
            2 => _mm512_castsi512_si128(_mm512_shuffle_i64x2(x, x, 0b_00_00_10_10)),
            3 => _mm512_castsi512_si128(_mm512_shuffle_i64x2(x, x, 0b_00_00_11_11)),
            _ => unreachable!(),
        }
    }

    #[inline(always)]
    pub unsafe fn xor2_x4(a: V512, b: V512) -> V512 { _mm512_xor_si512(a, b) }

    #[inline(always)]
    pub unsafe fn xor3_x4(a: V512, b: V512, c: V512) -> V512 {
        _mm512_ternarylogic_epi64(a, b, c, 0x96)
    }

    #[inline(always)]
    pub unsafe fn and_x4(a: V512, b: V512) -> V512 { _mm512_and_si512(a, b) }

    #[inline(always)]
    pub unsafe fn shuffle_x4(x: V512, sh: V128) -> V512 {
        _mm512_shuffle_epi8(x, broadcast(sh))
    }

    #[inline(always)]
    pub unsafe fn aesenc_x4(state: V512, rk: V128) -> V512 {
        _mm512_aesenc_epi128(state, broadcast(rk))
    }

    #[inline(always)]
    pub unsafe fn aesdec_x4(state: V512, rk: V128) -> V512 {
        _mm512_aesdec_epi128(state, broadcast(rk))
    }

    #[inline(always)]
    pub unsafe fn aesenclast_x4(state: V512, rk: V128) -> V512 {
        _mm512_aesenclast_epi128(state, broadcast(rk))
    }

    #[inline(always)]
    pub unsafe fn aesdeclast_x4(state: V512, rk: V128) -> V512 {
        _mm512_aesdeclast_epi128(state, broadcast(rk))
    }

    #[inline(always)]
    pub unsafe fn clmul_lo_x4(a: V512, b: V128) -> V512 {
        _mm512_clmulepi64_epi128(a, broadcast(b), 0x00)
    }

    #[inline(always)]
    pub unsafe fn bslli_3_x4(x: V512) -> V512 { _mm512_bslli_epi128::<3>(x) }

    #[inline(always)]
    pub unsafe fn blend_x4(a: V512, b: V512, mask: V512) -> V512 {
        // Byte-wise blend using ternary logic: (mask & b) | (~mask & a) = 0xCA
        _mm512_ternarylogic_epi64(mask, b, a, 0xCA)
    }

    #[inline(always)]
    pub unsafe fn splat8_x4(v: u8) -> V512 { _mm512_set1_epi8(v as i8) }

    /// andnot_x4: (~a) & b
    #[inline(always)]
    pub unsafe fn andnot_x4(a: V512, b: V512) -> V512 { _mm512_andnot_si512(a, b) }

    #[inline(always)]
    pub unsafe fn slli_epi64_32_x4(x: V512) -> V512 { _mm512_slli_epi64(x, 32) }
}
