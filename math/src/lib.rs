#![no_implicit_prelude]
#![cfg_attr(not(debug_assertions), allow(dead_code, unused_macros))]

mod simd;

use ::std::cmp::PartialEq;
use ::std::convert::From;
use ::std::default::Default;
use ::std::fmt::Debug;
use ::std::ops::{Add, AddAssign, Div, DivAssign, Index, Mul, Neg, Sub};

use simd::F32x4;

#[derive(Default, Copy, Clone, PartialEq, Debug)]
pub struct Vec3(F32x4);

impl Vec3 {
    #[inline(always)]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Vec3(F32x4::set(x, y, z, 0.0))
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
        // self.0.dot_product(other.0)

        let p = self.0 * other.0;
        // TODO: horizontal sum intrinsic?
        // let p = p.horizontal_add(p);
        // let p = p.horizontal_add(p);
        // p[0]
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

    #[inline(always)]
    pub fn sqrt(self) -> Vec3 {
        Vec3(self.0.sqrt())
    }

    #[inline(always)]
    pub fn x(self) -> f32 {
        self.0[0]
    }

    #[inline(always)]
    pub fn y(self) -> f32 {
        self.0[1]
    }

    #[inline(always)]
    pub fn z(self) -> f32 {
        self.0[2]
    }

    #[inline(always)]
    pub fn as_u8(self) -> (u8, u8, u8) {
        let a = self * 255.99;
        (a[0] as u8, a[1] as u8, a[2] as u8)
    }
}

impl From<Point3> for Vec3 {
    #[inline(always)]
    fn from(p: Point3) -> Self {
        Vec3(p.0)
    }
}

impl Index<usize> for Vec3 {
    type Output = f32;

    #[inline(always)]
    fn index(&self, i: usize) -> &f32 {
        &self.0[i]
    }
}

impl Add for Vec3 {
    type Output = Vec3;

    #[inline(always)]
    fn add(self, other: Vec3) -> Vec3 {
        Vec3(self.0 + other.0)
    }
}

impl AddAssign for Vec3 {
    #[inline(always)]
    fn add_assign(&mut self, other: Vec3) {
        self.0 += other.0
    }
}

impl Sub for Vec3 {
    type Output = Vec3;

    #[inline(always)]
    fn sub(self, other: Vec3) -> Vec3 {
        Vec3(self.0 - other.0)
    }
}

impl Mul for Vec3 {
    type Output = Vec3;

    #[inline(always)]
    fn mul(self, other: Vec3) -> Vec3 {
        Vec3(self.0 * other.0)
    }
}

impl Mul<f32> for Vec3 {
    type Output = Vec3;

    #[inline(always)]
    fn mul(self, s: f32) -> Vec3 {
        Vec3(self.0 * s)
    }
}

impl Mul<Vec3> for f32 {
    type Output = Vec3;

    #[inline(always)]
    fn mul(self, v: Vec3) -> Vec3 {
        Vec3(v.0 * self)
    }
}

impl Div<f32> for Vec3 {
    type Output = Vec3;

    #[inline(always)]
    fn div(self, s: f32) -> Vec3 {
        Vec3(self.0 / s)
    }
}

impl DivAssign<f32> for Vec3 {
    #[inline(always)]
    fn div_assign(&mut self, s: f32) {
        self.0 /= s
    }
}

impl Neg for Vec3 {
    type Output = Vec3;

    #[inline(always)]
    fn neg(self) -> Vec3 {
        Vec3(-self.0)
    }
}

#[derive(Default, Copy, Clone, PartialEq, Debug)]
pub struct Point3(F32x4);

impl Point3 {
    #[inline(always)]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Point3(F32x4::set(x, y, z, 0.0))
    }

    #[inline(always)]
    pub fn x(self) -> f32 {
        self.0[0]
    }

    #[inline(always)]
    pub fn y(self) -> f32 {
        self.0[1]
    }

    #[inline(always)]
    pub fn z(self) -> f32 {
        self.0[2]
    }

    #[inline(always)]
    pub fn min(self, other: Point3) -> Point3 {
        Point3(self.0.min(other.0))
    }

    #[inline(always)]
    pub fn max(self, other: Point3) -> Point3 {
        Point3(self.0.max(other.0))
    }
}

impl Index<usize> for Point3 {
    type Output = f32;

    #[inline(always)]
    fn index(&self, i: usize) -> &f32 {
        &self.0[i]
    }
}

impl Add<Vec3> for Point3 {
    type Output = Point3;

    #[inline(always)]
    fn add(self, v: Vec3) -> Point3 {
        Point3(self.0 + v.0)
    }
}

impl Sub for Point3 {
    type Output = Vec3;

    #[inline(always)]
    fn sub(self, other: Point3) -> Vec3 {
        Vec3(self.0 - other.0)
    }
}

impl Sub<Vec3> for Point3 {
    type Output = Point3;

    #[inline(always)]
    fn sub(self, v: Vec3) -> Point3 {
        Point3(self.0 - v.0)
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
