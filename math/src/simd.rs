#[cfg(target_arch = "x86")]
use ::std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use ::std::arch::x86_64::*;
//use ::std::clone::Clone;
use ::std::cmp::PartialEq;
use ::std::default::Default;
use ::std::fmt::{Debug, Formatter, Result};
use ::std::mem::transmute;
use ::std::ops::{Add, AddAssign, Div, DivAssign, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug)]
pub struct Components {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

#[derive(Clone, Copy)]
#[repr(align(16))]
pub union Vec3 {
    r: __m128,
    c: Components,
    a: [f32; 4],
}

impl Debug for Vec3 {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_tuple("Vec3").field(unsafe { &self.r }).finish()
    }
}

impl Vec3 {
    #[inline(always)]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Vec3 {
            r: unsafe { _mm_set_ps(0.0, z, y, x) },
        }
    }

    #[inline(always)]
    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    #[inline(always)]
    pub fn unit(self) -> Vec3 {
        // div by zero -> panic
        self / self.length()
    }

    #[inline(always)]
    pub fn dot(self, other: Vec3) -> f32 {
        // _mm_dp_ps() is slower.
        // let dp = Vec3(unsafe { _mm_dp_ps(self.0, other.0, 0b01110001) });
        // dp.x()
        let p = self * other;
        // let a = p.as_array();
        unsafe { p.c.x + p.c.y + p.c.z }
    }

    #[inline(never)]
    pub fn cross(&self, other: &Vec3) -> Vec3 {
        // XXX opt
        Vec3::new(
            self.y() * other.z() - self.z() * other.y(),
            self.z() * other.x() - self.x() * other.z(),
            self.x() * other.y() - self.y() * other.x(),
        )
    }

    #[inline(always)]
    pub fn sqrt(self) -> Vec3 {
        Vec3 {
            r: unsafe { _mm_sqrt_ps(self.r) },
        }
    }

    #[inline(always)]
    pub fn min(self, other: Vec3) -> Vec3 {
        Vec3 {
            r: unsafe { _mm_min_ps(self.r, other.r) },
        }
    }

    #[inline(always)]
    pub fn max(self, other: Vec3) -> Vec3 {
        Vec3 {
            r: unsafe { _mm_max_ps(self.r, other.r) },
        }
    }

    #[inline(always)]
    pub fn x(self) -> f32 {
        unsafe { self.c.x }
    }

    #[inline(always)]
    pub fn y(self) -> f32 {
        unsafe { self.c.y }
    }

    #[inline(always)]
    pub fn z(self) -> f32 {
        unsafe { self.c.z }
    }

    #[inline(always)]
    fn w(self) -> f32 {
        unsafe { self.c.w }
    }

    #[inline(always)]
    pub fn as_array(&self) -> [f32; 4] {
        unsafe { self.a }
    }

    #[inline(always)]
    pub fn as_u8(self) -> (u8, u8, u8) {
        let a = self * 255.99;
        (a.x() as u8, a.y() as u8, a.z() as u8)
    }

    #[inline(always)]
    pub fn reciprocal(&self) -> Vec3 {
        Vec3 {
            r: unsafe { _mm_rcp_ps(self.r) },
        }
    }

    // #[inline(always)]
    // pub fn sin(&self) -> Vec3 {
    //     Vec3(unsafe { _mm_sin_ps(self.0) })
    // }

    pub fn simd_cmpgt(self, other: Vec3) -> Vec3 {
        Vec3 {
            r: unsafe { _mm_cmpgt_ps(self.r, other.r) },
        }
    }
}

impl Default for Vec3 {
    #[inline(always)]
    fn default() -> Vec3 {
        Vec3 {
            r: unsafe { _mm_setzero_ps() },
        }
    }
}

// ===== Add =====

impl Add for Vec3 {
    type Output = Vec3;

    #[inline(always)]
    fn add(self, other: Vec3) -> Vec3 {
        Vec3 {
            r: unsafe { _mm_add_ps(self.r, other.r) },
        }
    }
}

// ===== AddAssign =====

impl AddAssign for Vec3 {
    #[inline(always)]
    fn add_assign(&mut self, other: Vec3) {
        self.r = unsafe { _mm_add_ps(self.r, other.r) };
    }
}

// ===== Sub =====

impl Sub for Vec3 {
    type Output = Vec3;

    #[inline(always)]
    fn sub(self, other: Vec3) -> Vec3 {
        Vec3 {
            r: unsafe { _mm_sub_ps(self.r, other.r) },
        }
    }
}

// ===== Neg =====

impl Neg for Vec3 {
    type Output = Vec3;

    #[inline(always)]
    fn neg(self) -> Vec3 {
        Vec3 {
            r: unsafe {
                _mm_xor_ps(self.r, _mm_set1_ps(f32::from_bits(0x80000000)))
            },
        }
        // Vec3(unsafe { _mm_mul_ps(self.0, _mm_set1_ps(-1.0)) })
    }
}

// ===== Mul =====

impl Mul for Vec3 {
    type Output = Vec3;

    #[inline(always)]
    fn mul(self, other: Vec3) -> Vec3 {
        Vec3 {
            r: unsafe { _mm_mul_ps(self.r, other.r) },
        }
    }
}

impl Mul<f32> for Vec3 {
    type Output = Vec3;

    #[inline(always)]
    fn mul(self, s: f32) -> Vec3 {
        Vec3 {
            r: unsafe { _mm_mul_ps(self.r, _mm_set1_ps(s)) },
        }
    }
}

impl Mul<Vec3> for f32 {
    type Output = Vec3;

    #[inline(always)]
    fn mul(self, v: Vec3) -> Vec3 {
        v * self
    }
}

// ===== Div =====

impl Div<f32> for Vec3 {
    type Output = Vec3;

    #[inline(always)]
    fn div(self, s: f32) -> Vec3 {
        Vec3 {
            r: unsafe { _mm_div_ps(self.r, _mm_set1_ps(s)) },
        }
    }
}

impl Div<Vec3> for f32 {
    type Output = Vec3;

    #[inline(always)]
    fn div(self, v: Vec3) -> Vec3 {
        Vec3 {
            r: unsafe { _mm_div_ps(_mm_set1_ps(self), v.r) },
        }
    }
}

impl Div<Vec3> for Vec3 {
    type Output = Vec3;

    #[inline(always)]
    fn div(self, other: Vec3) -> Vec3 {
        // _mm_mask_div_ps 0x7
        Vec3 {
            r: unsafe { _mm_div_ps(self.r, other.r) },
        }
    }
}

// ===== DivAssign =====

impl DivAssign<f32> for Vec3 {
    #[inline(always)]
    fn div_assign(&mut self, s: f32) {
        self.r = unsafe { _mm_div_ps(self.r, _mm_set1_ps(s)) };
    }
}

impl PartialEq for Vec3 {
    #[inline(always)]
    fn eq(&self, other: &Vec3) -> bool {
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
        assert_eq!(v.w(), 0.0);
    }

    #[test]
    fn test_vec3_add() {
        let v1 = Vec3::new(1.0, 0.0, 0.0);
        let v2 = Vec3::new(0.0, 0.0, -1.0);
        let v3 = v1 + v2;
        assert_eq!(v3.x(), 1.0);
        assert_eq!(v3.y(), 0.0);
        assert_eq!(v3.z(), -1.0);
        assert_eq!(v3.w(), 0.0);
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
        assert_eq!(v1.dot(v2), (1 * 2 + -5 * 3 + 9 * -4) as f32);
    }
}
