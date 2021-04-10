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

#[derive(Default, Copy, Clone, PartialEq, Debug)]
pub struct Vec3(F32x4);

impl Vec3 {
    #[inline]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Vec3(F32x4::new(x, y, z, 0.0))
    }

    #[inline]
    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    #[inline]
    pub fn unit(self) -> Vec3 {
        // div by zero -> panic
        self / self.length()
    }

    #[inline]
    pub fn dot(self, other: Vec3) -> f32 {
        // _mm_dp_ps() is slower.
        // let dp = Vec3(unsafe { _mm_dp_ps(self.0, other.0, 0b01110001) });
        // dp.x()
        let p = self.0 * other.0;
        // let a = p.as_array();
        // TODO: horizontal sum intrinsic?
        p[0] + p[1] + p[2] + p[3]
    }

    pub fn cross(&self, other: &Vec3) -> Vec3 {
        // TODO: use intrinsics
        Vec3::new(
            self.y() * other.z() - self.z() * other.y(),
            self.z() * other.x() - self.x() * other.z(),
            self.x() * other.y() - self.y() * other.x(),
        )
    }

    #[inline]
    pub fn min(self, other: Vec3) -> Vec3 {
        Vec3(self.0.min(other.0))
    }

    #[inline]
    pub fn max(self, other: Vec3) -> Vec3 {
        Vec3(self.0.max(other.0))
    }

    #[inline]
    pub fn sqrt(self) -> Vec3 {
        Vec3(self.0.sqrt())
    }

    #[inline]
    pub fn x(self) -> f32 {
        self.0[0]
    }

    #[inline]
    pub fn y(self) -> f32 {
        self.0[1]
    }

    #[inline]
    pub fn z(self) -> f32 {
        self.0[2]
    }

    #[inline]
    pub fn as_u8(self) -> (u8, u8, u8) {
        let a = self * 255.99;
        (a[0] as u8, a[1] as u8, a[2] as u8)
    }
}

impl Index<usize> for Vec3 {
    type Output = f32;

    #[inline]
    fn index(&self, i: usize) -> &f32 {
        &self.0[i]
    }
}

impl Add for Vec3 {
    type Output = Vec3;

    #[inline]
    fn add(self, other: Vec3) -> Vec3 {
        Vec3(self.0 + other.0)
    }
}

impl AddAssign for Vec3 {
    #[inline]
    fn add_assign(&mut self, other: Vec3) {
        self.0 += other.0
    }
}

impl Sub for Vec3 {
    type Output = Vec3;

    #[inline]
    fn sub(self, other: Vec3) -> Vec3 {
        Vec3(self.0 - other.0)
    }
}

impl Mul for Vec3 {
    type Output = Vec3;

    #[inline]
    fn mul(self, other: Vec3) -> Vec3 {
        Vec3(self.0 * other.0)
    }
}

impl Mul<f32> for Vec3 {
    type Output = Vec3;

    #[inline]
    fn mul(self, s: f32) -> Vec3 {
        Vec3(self.0 * s)
    }
}

impl Mul<Vec3> for f32 {
    type Output = Vec3;

    #[inline]
    fn mul(self, v: Vec3) -> Vec3 {
        Vec3(v.0 * self)
    }
}

impl Div<f32> for Vec3 {
    type Output = Vec3;

    #[inline]
    fn div(self, s: f32) -> Vec3 {
        Vec3(self.0 / s)
    }
}

impl DivAssign<f32> for Vec3 {
    #[inline]
    fn div_assign(&mut self, s: f32) {
        self.0 /= s
    }
}

impl Neg for Vec3 {
    type Output = Vec3;

    #[inline]
    fn neg(self) -> Vec3 {
        Vec3(-self.0)
    }
}

#[derive(Clone, Copy)]
#[repr(align(16))] // TODO: switch to repr(simd).
union F32x4 {
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
    fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
        F32x4 {
            r: unsafe { _mm_set_ps(d, c, b, a) },
        }
    }

    #[inline(always)]
    fn sqrt(self) -> F32x4 {
        F32x4 {
            r: unsafe { _mm_sqrt_ps(self.r) },
        }
    }

    #[inline(always)]
    fn min(self, other: F32x4) -> F32x4 {
        F32x4 {
            r: unsafe { _mm_min_ps(self.r, other.r) },
        }
    }

    #[inline(always)]
    fn max(self, other: F32x4) -> F32x4 {
        F32x4 {
            r: unsafe { _mm_max_ps(self.r, other.r) },
        }
    }

    #[inline(always)]
    fn as_array(&self) -> [f32; 4] {
        unsafe { self.a }
    }

    #[inline(always)]
    fn reciprocal(&self) -> F32x4 {
        F32x4 {
            r: unsafe { _mm_rcp_ps(self.r) },
        }
    }

    // #[inline(always)]
    // fn sin(&self) -> Vec3 {
    //     Vec3(unsafe { _mm_sin_ps(self.0) })
    // }

    fn simd_cmpgt(self, other: F32x4) -> F32x4 {
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

    #[inline(always)]
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
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(v.x(), 1.0);
        assert_eq!(v.y(), 2.0);
        assert_eq!(v.z(), 3.0);
        assert_eq!(v.0[3], 0.0);
    }

    #[test]
    fn test_vec3_add() {
        let v1 = Vec3::new(1.0, 0.0, 0.0);
        let v2 = Vec3::new(0.0, 0.0, -1.0);
        let v3 = v1 + v2;
        assert_eq!(v3.x(), 1.0);
        assert_eq!(v3.y(), 0.0);
        assert_eq!(v3.z(), -1.0);
    }

    #[test]
    fn test_vec3_neg() {
        let v1 = Vec3::new(-1.2, 3.4, -5.6789);
        let v2 = Vec3::new(1.2, -3.4, 5.6789);
        assert_eq!(-v1, v2);
    }

    #[test]
    fn test_vec3_dot() {
        let v1 = Vec3::new(1.0, -5.0, 9.0);
        let v2 = Vec3::new(2.0, 3.0, -4.0);
        assert_eq!(v1.dot(v2), 1.0 * 2.0 + -5.0 * 3.0 + 9.0 * -4.0);
    }
}
