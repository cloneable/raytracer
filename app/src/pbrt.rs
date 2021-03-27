use ::std::boxed::Box;
use ::std::clone::Clone;
use ::std::cmp::Ordering;
use ::std::default::Default;
use ::std::fmt::Debug;
use ::std::iter::Iterator;
use ::std::option::Option::{self, None, Some};
use ::std::rc::Rc;
use ::std::unreachable;
use ::std::vec::Vec;

// use ::math::Vec3;
use ::math::simd::Vec3;
use ::rand::Rng;
use ::rand::SeedableRng;

#[derive(Default, Clone)]
pub struct HitRecord {
    pub t: f32,
    pub p: Vec3,
    pub normal: Vec3,
    pub material: usize,
    pub u: f32,
    pub v: f32,
}

pub trait Hitable: Debug {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32, rec: &mut HitRecord)
        -> bool;
    fn bounding_box(&self, t0: f32, t1: f32) -> Option<AABB>;
}

#[derive(Default)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
    pub time: f32,
    pub max_depth: usize,
}

impl Ray {
    pub fn new(
        origin: Vec3, direction: Vec3, time: f32, max_depth: usize,
    ) -> Self {
        Ray {
            origin,
            direction: direction.unit(),
            time,
            max_depth,
        }
    }

    pub fn point_at_param(&self, t: f32) -> Vec3 {
        self.origin + t * self.direction
    }

    pub fn trace(
        &self, rng: &mut RNG, world: &dyn Hitable, matlib: &MaterialLibrary,
        depth: usize,
    ) -> Vec3 {
        let mut rec = HitRecord::default();
        if world.hit(self, 0.001, f32::MAX, &mut rec) {
            let mut scattered = Ray::default();
            let mut attenuation = Vec3::default();
            let mat = &matlib.lib[rec.material];
            let emitted = mat.emitted(rec.u, rec.v, rec.p);
            if depth < self.max_depth
                && mat.scatter(
                    rng,
                    self,
                    &mut rec,
                    &mut attenuation,
                    &mut scattered,
                )
            {
                return emitted
                    + attenuation
                        * scattered.trace(rng, world, matlib, depth + 1);
            }
            return emitted;
        }
        // let t = 0.5 * (self.direction.y() + 1.0);
        // (1.0 - t) * Vec3::new(1.0, 1.0, 1.0) + t * Vec3::new(0.5, 0.7, 1.0)
        return Vec3::default();
    }
}

pub struct RNG {
    rng: Box<dyn ::rand::RngCore>,
}

impl RNG {
    pub fn rand(&mut self) -> f32 {
        self.rng.gen::<f32>()
        //::rand::random::<f32>()
    }

    pub fn random_in_unit_sphere(&mut self) -> Vec3 {
        loop {
            let v = 2.0 * Vec3::new(self.rand(), self.rand(), self.rand())
                - Vec3::new(1.0, 1.0, 1.0);
            if v.dot(v) >= 1.0 {
                return v;
            }
            // TODO: count misses
        }
    }

    pub fn random_in_unit_disk(&mut self) -> Vec3 {
        loop {
            let v = 2.0 * Vec3::new(self.rand(), self.rand(), 0.0)
                - Vec3::new(1.0, 1.0, 0.0);
            if v.dot(v) < 1.0 {
                return v;
            }
            // TODO: count misses
        }
    }
}

impl Default for RNG {
    fn default() -> Self {
        RNG {
            rng: Box::new(::rand::prelude::SmallRng::from_seed([
                4, 2, 3, 4, 5, 6, 7, 8, //br
                9, 10, 11, 12, 13, 14, 15, 16, //br
                17, 18, 19, 20, 21, 22, 23, 24, //br
                25, 26, 27, 28, 29, 30, 31, 32,
            ])),
        }
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct AABB {
    min: Vec3,
    max: Vec3,
}

impl AABB {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        AABB { min, max }
    }

    pub fn surround(b0: AABB, b1: AABB) -> Self {
        AABB {
            min: b0.min.min(b1.min),
            max: b0.max.max(b1.max),
        }
    }

    #[inline(never)]
    pub fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> bool {
        let r_origin = r.origin.as_array();
        let r_direction = r.direction.as_array();
        let min = self.min.as_array();
        let max = self.max.as_array();
        for a in 0..=2 {
            // TODO: simd this.
            let inv_dir = 1.0 / r_direction[a];
            let mut t0 = (min[a] - r_origin[a]) * inv_dir;
            let mut t1 = (max[a] - r_origin[a]) * inv_dir;
            if inv_dir < 0.0 {
                ::std::mem::swap(&mut t0, &mut t1);
            }
            let tmin = if t0 > t_min { t0 } else { t_min };
            let tmax = if t1 < t_max { t1 } else { t_max };
            if tmin > tmax {
                return false;
            }
        }

        // let inv_dir = r.direction.reciprocal();
        // let t0 = (self.min - r.origin) * inv_dir;
        // let t1 = (self.max - r.origin) * inv_dir;
        // // inv swap
        //  check with _mm_movemask_ps or _mm_fpclass_ps_mask?
        // let tmin = t0.max(Vec3::new(t_min, t_min, t_min));
        // let tmax = t1.min(Vec3::new(t_max, t_max, t_max));
        // // tmin > tmax

        return true;
    }
}

#[derive(Default, Clone, Debug)]
pub struct HitableList {
    pub list: Vec<Rc<dyn Hitable>>,
}

impl HitableList {
    pub fn partition(self, index: usize) -> (Self, Self) {
        let mut left = Vec::with_capacity(self.list.len());
        let mut right = Vec::with_capacity(self.list.len());
        let mut i = 0;
        while i < self.list.len() {
            if i < index {
                left.push(self.list[i].clone());
            } else {
                right.push(self.list[i].clone());
            }
            i += 1;
        }
        (HitableList { list: left }, HitableList { list: right })
    }
}

impl Hitable for HitableList {
    fn hit(
        &self, r: &Ray, t_min: f32, t_max: f32, rec: &mut HitRecord,
    ) -> bool {
        let mut temp_rec = HitRecord::default();
        let mut hit_anything = false;
        let mut closest_so_far = t_max;
        for e in &self.list {
            if e.hit(r, t_min, closest_so_far, &mut temp_rec) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                *rec = temp_rec.clone();
            }
        }
        return hit_anything;
    }

    fn bounding_box(&self, t0: f32, t1: f32) -> Option<AABB> {
        if self.list.is_empty() {
            return None;
        }
        let first = self.list.first().unwrap();
        if let Some(bb) = first.bounding_box(t0, t1) {
            return Some(self.list.iter().skip(1).fold(bb, |a, b| {
                if let Some(bbox) = b.bounding_box(t0, t1) {
                    AABB::surround(a, bbox)
                } else {
                    // TODO: handle no-bb case
                    a
                }
            }));
        }
        unreachable!()
    }
}

#[derive(Debug)]
pub struct BVH {
    bb: AABB,
    left: Rc<dyn Hitable>,
    right: Rc<dyn Hitable>,
}

fn box_x_compare(a: &Rc<dyn Hitable>, b: &Rc<dyn Hitable>) -> Ordering {
    if let Some(abox) = a.bounding_box(0.0, 0.0) {
        if let Some(bbox) = b.bounding_box(0.0, 0.0) {
            if (abox.min.x() - bbox.min.x()) < 0.0 {
                return Ordering::Less;
            }
            return Ordering::Greater;
        }
    }
    unreachable!()
}

fn box_y_compare<'a, 'b>(a: &Rc<dyn Hitable>, b: &Rc<dyn Hitable>) -> Ordering {
    if let Some(abox) = a.bounding_box(0.0, 0.0) {
        if let Some(bbox) = b.bounding_box(0.0, 0.0) {
            if (abox.min.y() - bbox.min.y()) < 0.0 {
                return Ordering::Less;
            }
            return Ordering::Greater;
        }
    }
    unreachable!()
}

fn box_z_compare(a: &Rc<dyn Hitable>, b: &Rc<dyn Hitable>) -> Ordering {
    if let Some(abox) = a.bounding_box(0.0, 0.0) {
        if let Some(bbox) = b.bounding_box(0.0, 0.0) {
            if (abox.min.z() - bbox.min.z()) < 0.0 {
                return Ordering::Less;
            }
            return Ordering::Greater;
        }
    }
    unreachable!()
}

impl BVH {
    pub fn new(hl: HitableList, t0: f32, t1: f32, rng: &mut RNG) -> Self {
        let axis = (3.0 * rng.rand()) as u8;
        let mut list = hl.clone();
        match axis {
            0 => list.list.sort_by(box_x_compare),
            1 => list.list.sort_by(box_y_compare),
            _ => list.list.sort_by(box_z_compare),
        };
        let left;
        let right;
        match list.list.len() {
            0 => {
                unreachable!()
            }
            1 => {
                left = list.list[0].clone();
                right = list.list[0].clone();
            }
            2 => {
                left = list.list[0].clone();
                right = list.list[1].clone();
            }
            _ => {
                let index = list.list.len() / 2;
                let (ll, lr) = list.partition(index);
                left = Rc::new(BVH::new(ll, t0, t1, rng));
                right = Rc::new(BVH::new(lr, t0, t1, rng));
            }
        };
        if let Some(abox) = left.bounding_box(t0, t1) {
            if let Some(bbox) = right.bounding_box(t0, t1) {
                return BVH {
                    bb: AABB::surround(abox, bbox),
                    left,
                    right,
                };
            }
        }
        unreachable!()
    }
}

impl Hitable for BVH {
    fn hit(
        &self, r: &Ray, t_min: f32, t_max: f32, rec: &mut HitRecord,
    ) -> bool {
        if !self.bb.hit(r, t_min, t_max) {
            return false;
        }
        let mut left_rec = HitRecord::default();
        let mut right_rec = HitRecord::default();
        let hit_left = self.left.hit(r, t_min, t_max, &mut left_rec);
        let hit_right = self.right.hit(r, t_min, t_max, &mut right_rec);
        if hit_left && hit_right {
            if left_rec.t < right_rec.t {
                *rec = left_rec;
            } else {
                *rec = right_rec;
            }
            return true;
        } else if hit_left {
            *rec = left_rec;
            return true;
        } else if hit_right {
            *rec = right_rec;
            return true;
        }
        return false;
    }

    fn bounding_box(&self, _: f32, _: f32) -> Option<AABB> {
        Some(self.bb)
    }
}

pub struct Camera {
    origin: Vec3,
    lower_left_corner: Vec3,
    horizontal: Vec3,
    vertical: Vec3,

    u: Vec3,
    v: Vec3,
    w: Vec3,

    lens_radius: f32,

    t0: f32,
    t1: f32,
}

impl Camera {
    pub fn new(
        look_from: Vec3, look_at: Vec3, up: &Vec3, vfov: f32, aspect: f32,
        aperture: f32, focus_dist: f32, t0: f32, t1: f32,
    ) -> Self {
        let lens_radius = aperture / 2.0;
        let theta = vfov.to_radians();
        let half_height = (theta / 2.0).tan();
        let half_width = aspect * half_height;
        let w = (look_from - look_at).unit();
        let u = up.cross(&w).unit();
        let v = w.cross(&u);
        Camera {
            origin: look_from,
            lower_left_corner: look_from
                - half_width * focus_dist * u
                - half_height * focus_dist * v
                - focus_dist * w,
            horizontal: 2.0 * half_width * focus_dist * u,
            vertical: 2.0 * half_height * focus_dist * v,
            u,
            v,
            w,
            lens_radius,
            t0,
            t1,
        }
    }

    pub fn get_ray(&self, rng: &mut RNG, u: f32, v: f32, depth: usize) -> Ray {
        let rd = self.lens_radius * rng.random_in_unit_disk();
        let offset = self.u * rd.x() + self.v * rd.y();
        let t = self.t0 + rng.rand() * (self.t1 - self.t0);
        Ray::new(
            self.origin + offset,
            self.lower_left_corner + u * self.horizontal + v * self.vertical
                - self.origin
                - offset,
            t,
            depth,
        )
    }
}

pub trait Material {
    fn scatter(
        &self, rng: &mut RNG, ray: &Ray, rec: &mut HitRecord,
        attenuation: &mut Vec3, scattered: &mut Ray,
    ) -> bool;

    fn emitted(&self, _u: f32, _v: f32, _p: Vec3) -> Vec3 {
        Vec3::default()
    }
}

#[derive(Default)]
pub struct MaterialLibrary {
    pub lib: Vec<Box<dyn Material>>,
}

pub trait Texture: Debug {
    fn value(&self, u: f32, v: f32, p: Vec3) -> Vec3;
}
