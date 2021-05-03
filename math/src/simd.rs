#[cfg(target_arch = "x86_64")]
use ::std::arch::x86_64::{
    __m128,
    _mm_add_ps,       // SSE
    _mm_cmpeq_ps,     // SSE
    _mm_cmpgt_ps,     // SSE
    _mm_div_ps,       // SSE
    _mm_dp_ps,        // SSE4_1
    _mm_hadd_ps,      // SSE3
    _mm_max_ps,       // SSE
    _mm_min_ps,       // SSE
    _mm_movemask_ps,  // SSE
    _mm_mul_ps,       // SSE
    _mm_rcp_ps,       // SSE
    _mm_set1_ps,      // SSE
    _mm_set_ps,       // SSE
    _mm_setzero_ps,   // SSE
    _mm_shuffle_ps,   // SSE
    _mm_sqrt_ps,      // SSE
    _mm_sub_ps,       // SSE
    _mm_undefined_ps, // SSE
    _mm_xor_ps,       // SSE
};
use ::std::cmp::PartialEq;
use ::std::default::Default;
use ::std::fmt::{Debug, Formatter, Result};
use ::std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub,
    SubAssign,
};

#[derive(Clone, Copy)]
#[repr(align(16))] // TODO: switch to repr(simd).
pub(super) union F32x4 {
    r: __m128,
    a: [f32; 4],
}

impl Debug for F32x4 {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_tuple("F32x4").field(unsafe { &self.r }).finish()
    }
}

impl F32x4 {
    #[inline(always)]
    #[must_use]
    pub fn set(a: f32, b: f32, c: f32, d: f32) -> Self {
        F32x4 {
            r: unsafe { _mm_set_ps(d, c, b, a) },
        }
    }

    #[inline(always)]
    #[must_use]
    pub fn set1(a: f32) -> Self {
        F32x4 {
            r: unsafe { _mm_set1_ps(a) },
        }
    }

    #[inline(always)]
    #[must_use]
    pub unsafe fn undefined() -> Self {
        F32x4 {
            r: _mm_undefined_ps(),
        }
    }

    #[inline(always)]
    #[must_use]
    pub fn sqrt(self) -> F32x4 {
        F32x4 {
            r: unsafe { _mm_sqrt_ps(self.r) },
        }
    }

    #[inline(always)]
    #[must_use]
    pub fn min(self, other: F32x4) -> F32x4 {
        F32x4 {
            r: unsafe { _mm_min_ps(self.r, other.r) },
        }
    }

    #[inline(always)]
    #[must_use]
    pub fn max(self, other: F32x4) -> F32x4 {
        F32x4 {
            r: unsafe { _mm_max_ps(self.r, other.r) },
        }
    }

    #[inline(always)]
    #[must_use]
    pub fn as_array(&self) -> [f32; 4] {
        unsafe { self.a }
    }

    #[inline(always)]
    #[must_use]
    pub fn reciprocal(&self) -> F32x4 {
        F32x4 {
            r: unsafe { _mm_rcp_ps(self.r) },
        }
    }

    // #[inline(always)]
    // fn sin(&self) -> Vec3 {
    //     Vec3(unsafe { _mm_sin_ps(self.0) })
    // }

    #[inline(always)]
    #[must_use]
    pub fn simd_cmpgt(self, other: F32x4) -> F32x4 {
        F32x4 {
            r: unsafe { _mm_cmpgt_ps(self.r, other.r) },
        }
    }

    #[inline(always)]
    #[must_use]
    pub fn dot_product(self, other: F32x4) -> f32 {
        // TODO: _mm_dp_ps() is slow. Because SSE4.1?
        let a = F32x4 {
            r: unsafe { _mm_dp_ps(self.r, other.r, 0b01110001) },
        };
        a[0]
    }

    #[inline(always)]
    #[must_use]
    pub fn horizontal_add(self, other: F32x4) -> F32x4 {
        F32x4 {
            r: unsafe { _mm_hadd_ps(self.r, other.r) },
        }
    }

    #[inline(always)]
    #[must_use]
    pub fn shuffle<const CTRL: u8>(self) -> F32x4 {
        F32x4 {
            r: unsafe { _mm_shuffle_ps(self.r, self.r, CTRL as i32) },
        }
    }
}

impl Default for F32x4 {
    #[inline(always)]
    #[must_use]
    fn default() -> F32x4 {
        F32x4 {
            r: unsafe { _mm_setzero_ps() },
        }
    }
}

impl Index<usize> for F32x4 {
    type Output = f32;

    #[inline(always)]
    #[must_use]
    fn index(&self, i: usize) -> &f32 {
        ::std::debug_assert!(i < 4);
        unsafe { self.a.get_unchecked(i) }
    }
}

impl IndexMut<usize> for F32x4 {
    #[inline(always)]
    #[must_use]
    fn index_mut(&mut self, i: usize) -> &mut f32 {
        ::std::debug_assert!(i < 4);
        unsafe { self.a.get_unchecked_mut(i) }
    }
}

impl Add for F32x4 {
    type Output = F32x4;

    #[inline(always)]
    #[must_use]
    fn add(self, other: F32x4) -> F32x4 {
        F32x4 {
            r: unsafe { _mm_add_ps(self.r, other.r) },
        }
    }
}

impl AddAssign for F32x4 {
    #[inline(always)]
    fn add_assign(&mut self, other: F32x4) {
        self.r = unsafe { _mm_add_ps(self.r, other.r) };
    }
}

impl Sub for F32x4 {
    type Output = F32x4;

    #[inline(always)]
    #[must_use]
    fn sub(self, other: F32x4) -> F32x4 {
        F32x4 {
            r: unsafe { _mm_sub_ps(self.r, other.r) },
        }
    }
}

impl SubAssign for F32x4 {
    #[inline(always)]
    fn sub_assign(&mut self, other: F32x4) {
        self.r = unsafe { _mm_sub_ps(self.r, other.r) };
    }
}

impl Neg for F32x4 {
    type Output = F32x4;

    #[inline(always)]
    #[must_use]
    fn neg(self) -> F32x4 {
        F32x4 {
            r: unsafe {
                _mm_xor_ps(self.r, _mm_set1_ps(f32::from_bits(0x80000000)))
            },
        }
    }
}

impl Mul for F32x4 {
    type Output = F32x4;

    #[inline(always)]
    #[must_use]
    fn mul(self, other: F32x4) -> F32x4 {
        F32x4 {
            r: unsafe { _mm_mul_ps(self.r, other.r) },
        }
    }
}

impl MulAssign for F32x4 {
    #[inline(always)]
    fn mul_assign(&mut self, other: F32x4) {
        self.r = unsafe { _mm_mul_ps(self.r, other.r) };
    }
}

impl Mul<f32> for F32x4 {
    type Output = F32x4;

    #[inline(always)]
    #[must_use]
    fn mul(self, s: f32) -> F32x4 {
        self * F32x4::set1(s)
    }
}

impl MulAssign<f32> for F32x4 {
    #[inline(always)]
    fn mul_assign(&mut self, s: f32) {
        *self *= F32x4::set1(s)
    }
}

impl Div<f32> for F32x4 {
    type Output = F32x4;

    #[inline(always)]
    #[must_use]
    fn div(self, s: f32) -> F32x4 {
        self / F32x4::set1(s)
    }
}

impl Div<F32x4> for F32x4 {
    type Output = F32x4;

    #[inline(always)]
    #[must_use]
    fn div(self, other: F32x4) -> F32x4 {
        // _mm_mask_div_ps 0x7
        F32x4 {
            r: unsafe { _mm_div_ps(self.r, other.r) },
        }
    }
}

impl DivAssign for F32x4 {
    #[inline(always)]
    fn div_assign(&mut self, other: F32x4) {
        self.r = unsafe { _mm_div_ps(self.r, other.r) };
    }
}

impl DivAssign<f32> for F32x4 {
    #[inline(always)]
    fn div_assign(&mut self, s: f32) {
        *self /= F32x4::set1(s)
    }
}

impl PartialEq for F32x4 {
    #[inline(always)]
    #[must_use]
    fn eq(&self, other: &F32x4) -> bool {
        // TODO: 0.0 != -0.0
        unsafe { _mm_movemask_ps(_mm_cmpeq_ps(self.r, other.r)) == 0 }
    }
}
