#[cfg(target_arch = "x86_64")]
use ::std::arch::x86_64::{
    __m128,
    _mm_add_ps,      // SSE
    _mm_cmpeq_ps,    // SSE
    _mm_cmpgt_ps,    // SSE
    _mm_div_ps,      // SSE
    _mm_max_ps,      // SSE
    _mm_min_ps,      // SSE
    _mm_movemask_ps, // SSE
    _mm_mul_ps,      // SSE
    _mm_rcp_ps,      // SSE
    _mm_set1_ps,     // SSE
    _mm_set_ps,      // SSE
    _mm_setzero_ps,  // SSE
    _mm_sqrt_ps,     // SSE
    _mm_sub_ps,      // SSE
    _mm_xor_ps,      // SSE
                     //_mm_dp_ps, // SSE4_1
};
use ::std::cmp::PartialEq;
use ::std::default::Default;
use ::std::fmt::{Debug, Formatter, Result};
use ::std::ops::{Add, AddAssign, Div, DivAssign, Index, Mul, Neg, Sub};

#[derive(Clone, Copy)]
#[repr(align(16))] // TODO: switch to repr(simd).
pub union F32x4 {
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
    pub fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
        F32x4 {
            r: unsafe { _mm_set_ps(d, c, b, a) },
        }
    }

    #[inline(always)]
    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    #[inline(always)]
    pub fn unit(self) -> F32x4 {
        // div by zero -> panic
        self / self.length()
    }

    #[inline(always)]
    pub fn dot(self, other: F32x4) -> f32 {
        // _mm_dp_ps() is slower.
        // let dp = Vec3(unsafe { _mm_dp_ps(self.0, other.0, 0b01110001) });
        // dp.x()
        let p = self * other;
        // let a = p.as_array();
        // TODO: horizontal sum intrinsic?
        p[0] + p[1] + p[2] + p[3]
    }

    #[inline(never)]
    pub fn cross(&self, other: &F32x4) -> F32x4 {
        // TODO: Move into new Vec3.
        // TODO: use intrinsics
        F32x4::new(
            self.y() * other.z() - self.z() * other.y(),
            self.z() * other.x() - self.x() * other.z(),
            self.x() * other.y() - self.y() * other.x(),
            0.0,
        )
    }

    #[inline(always)]
    pub fn sqrt(self) -> F32x4 {
        F32x4 {
            r: unsafe { _mm_sqrt_ps(self.r) },
        }
    }

    #[inline(always)]
    pub fn min(self, other: F32x4) -> F32x4 {
        F32x4 {
            r: unsafe { _mm_min_ps(self.r, other.r) },
        }
    }

    #[inline(always)]
    pub fn max(self, other: F32x4) -> F32x4 {
        F32x4 {
            r: unsafe { _mm_max_ps(self.r, other.r) },
        }
    }

    #[inline(always)]
    pub fn x(self) -> f32 {
        self[0]
    }

    #[inline(always)]
    pub fn y(self) -> f32 {
        self[1]
    }

    #[inline(always)]
    pub fn z(self) -> f32 {
        self[2]
    }

    #[inline(always)]
    pub fn w(self) -> f32 {
        self[3]
    }

    #[inline(always)]
    pub fn as_array(&self) -> [f32; 4] {
        unsafe { self.a }
    }

    #[inline(always)]
    pub fn as_u8(self) -> (u8, u8, u8, u8) {
        let a = self * 255.99;
        (a.x() as u8, a.y() as u8, a.z() as u8, a.w() as u8)
    }

    #[inline(always)]
    pub fn reciprocal(&self) -> F32x4 {
        F32x4 {
            r: unsafe { _mm_rcp_ps(self.r) },
        }
    }

    // #[inline(always)]
    // pub fn sin(&self) -> Vec3 {
    //     Vec3(unsafe { _mm_sin_ps(self.0) })
    // }

    pub fn simd_cmpgt(self, other: F32x4) -> F32x4 {
        F32x4 {
            r: unsafe { _mm_cmpgt_ps(self.r, other.r) },
        }
    }
}

impl Default for F32x4 {
    #[inline(always)]
    fn default() -> F32x4 {
        F32x4 {
            r: unsafe { _mm_setzero_ps() },
        }
    }
}

impl Index<usize> for F32x4 {
    type Output = f32;

    fn index(&self, i: usize) -> &f32 {
        ::std::debug_assert!(i < 4);
        unsafe { self.a.get_unchecked(i) }
    }
}

// ===== Add =====

impl Add for F32x4 {
    type Output = F32x4;

    #[inline(always)]
    fn add(self, other: F32x4) -> F32x4 {
        F32x4 {
            r: unsafe { _mm_add_ps(self.r, other.r) },
        }
    }
}

// ===== AddAssign =====

impl AddAssign for F32x4 {
    #[inline(always)]
    fn add_assign(&mut self, other: F32x4) {
        self.r = unsafe { _mm_add_ps(self.r, other.r) };
    }
}

// ===== Sub =====

impl Sub for F32x4 {
    type Output = F32x4;

    #[inline(always)]
    fn sub(self, other: F32x4) -> F32x4 {
        F32x4 {
            r: unsafe { _mm_sub_ps(self.r, other.r) },
        }
    }
}

// ===== Neg =====

impl Neg for F32x4 {
    type Output = F32x4;

    #[inline(always)]
    fn neg(self) -> F32x4 {
        F32x4 {
            r: unsafe {
                _mm_xor_ps(self.r, _mm_set1_ps(f32::from_bits(0x80000000)))
            },
        }
        // Vec3(unsafe { _mm_mul_ps(self.0, _mm_set1_ps(-1.0)) })
    }
}

// ===== Mul =====

impl Mul for F32x4 {
    type Output = F32x4;

    #[inline(always)]
    fn mul(self, other: F32x4) -> F32x4 {
        F32x4 {
            r: unsafe { _mm_mul_ps(self.r, other.r) },
        }
    }
}

impl Mul<f32> for F32x4 {
    type Output = F32x4;

    #[inline(always)]
    fn mul(self, s: f32) -> F32x4 {
        F32x4 {
            r: unsafe { _mm_mul_ps(self.r, _mm_set1_ps(s)) },
        }
    }
}

impl Mul<F32x4> for f32 {
    type Output = F32x4;

    #[inline(always)]
    fn mul(self, v: F32x4) -> F32x4 {
        v * self
    }
}

// ===== Div =====

impl Div<f32> for F32x4 {
    type Output = F32x4;

    #[inline(always)]
    fn div(self, s: f32) -> F32x4 {
        F32x4 {
            r: unsafe { _mm_div_ps(self.r, _mm_set1_ps(s)) },
        }
    }
}

impl Div<F32x4> for f32 {
    type Output = F32x4;

    #[inline(always)]
    fn div(self, v: F32x4) -> F32x4 {
        F32x4 {
            r: unsafe { _mm_div_ps(_mm_set1_ps(self), v.r) },
        }
    }
}

impl Div<F32x4> for F32x4 {
    type Output = F32x4;

    #[inline(always)]
    fn div(self, other: F32x4) -> F32x4 {
        // _mm_mask_div_ps 0x7
        F32x4 {
            r: unsafe { _mm_div_ps(self.r, other.r) },
        }
    }
}

// ===== DivAssign =====

impl DivAssign<f32> for F32x4 {
    #[inline(always)]
    fn div_assign(&mut self, s: f32) {
        self.r = unsafe { _mm_div_ps(self.r, _mm_set1_ps(s)) };
    }
}

impl PartialEq for F32x4 {
    #[inline(always)]
    fn eq(&self, other: &F32x4) -> bool {
        unsafe { _mm_movemask_ps(_mm_cmpeq_ps(self.r, other.r)) == 0 }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use ::std::assert_eq;

    #[test]
    fn test_vec3_new() {
        let v = F32x4::new(1.0, 2.0, 3.0, 0.0);
        assert_eq!(v.x(), 1.0);
        assert_eq!(v.y(), 2.0);
        assert_eq!(v.z(), 3.0);
        assert_eq!(v.w(), 0.0);
    }

    #[test]
    fn test_vec3_add() {
        let v1 = F32x4::new(1.0, 0.0, 0.0, 0.0);
        let v2 = F32x4::new(0.0, 0.0, -1.0, 0.0);
        let v3 = v1 + v2;
        assert_eq!(v3.x(), 1.0);
        assert_eq!(v3.y(), 0.0);
        assert_eq!(v3.z(), -1.0);
        assert_eq!(v3.w(), 0.0);
    }

    #[test]
    fn test_vec3_neg() {
        let v1 = F32x4::new(-1.2, 3.4, -5.6789, 1.0);
        let v2 = F32x4::new(1.2, -3.4, 5.6789, 1.0);
        assert_eq!(-v1, v2);
    }

    #[test]
    fn test_vec3_dot() {
        let v1 = F32x4::new(1.0, -5.0, 9.0, 0.5);
        let v2 = F32x4::new(2.0, 3.0, -4.0, 0.5);
        assert_eq!(v1.dot(v2), 1.0 * 2.0 + -5.0 * 3.0 + 9.0 * -4.0 + 0.5 * 0.5);
    }
}
