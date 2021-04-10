#![no_implicit_prelude]
#![cfg_attr(not(debug_assertions), allow(dead_code, unused_macros))]

use ::std::boxed::Box;
use ::std::clone::Clone;
use ::std::default::Default;
use ::std::io::{self, Write};
use ::std::iter::Iterator;
use ::std::option::Option::{self, None, Some};
use ::std::print;
use ::std::rc::Rc;

// use ::math::Vec3;
use ::math::simd::F32x4;

type Vec3 = F32x4;

mod executor;
mod pbrt;

use pbrt::{
    Camera, HitRecord, Hitable, HitableList, Material, MaterialLibrary, Ray,
    Texture, AABB, BVH, RNG,
};

#[derive(Debug)]
struct Sphere {
    center: Vec3,
    radius: f32,
    radius_squared: f32,
    material: usize,
}

impl Sphere {
    fn new(center: Vec3, radius: f32, material: usize) -> Self {
        Sphere {
            center,
            radius,
            radius_squared: radius * radius,
            material,
        }
    }
}

const PI: f32 = ::std::f32::consts::PI;

impl Hitable for Sphere {
    fn hit(
        &self, r: &Ray, t_min: f32, t_max: f32, rec: &mut HitRecord,
    ) -> bool {
        let oc = r.origin - self.center;
        let a = r.direction.dot(r.direction);
        let b = oc.dot(r.direction);
        let c = oc.dot(oc) - self.radius_squared;
        let discriminat = b * b - a * c;
        if discriminat > 0.0 {
            let dsqrt = discriminat.sqrt();
            let temp = (-b - dsqrt) / a;
            let rec_p = r.point_at_param(temp);
            let normal = (rec_p - self.center) / self.radius;
            let phi = normal.z().atan2(normal.x());
            let theta = normal.y().asin();
            if temp < t_max && temp > t_min {
                rec.t = temp;
                rec.p = rec_p;
                rec.normal = normal;
                rec.material = self.material;
                rec.u = 1.0 - (phi + PI) / (2.0 * PI);
                rec.v = (theta - PI / 2.0) / PI;
                return true;
            }
            let temp = (-b + dsqrt) / a;
            if temp < t_max && temp > t_min {
                rec.t = temp;
                rec.p = rec_p;
                rec.normal = normal;
                rec.material = self.material;
                rec.u = 1.0 - (phi + PI) / (2.0 * PI);
                rec.v = (theta - PI / 2.0) / PI;
                return true;
            }
        }
        return false;
    }

    fn bounding_box(&self, _: f32, _: f32) -> Option<AABB> {
        let rv = Vec3::new(self.radius, self.radius, self.radius, 0.0);
        Some(AABB::new(self.center - rv, self.center + rv))
    }
}

#[derive(Debug)]
struct MovingSphere {
    center0: Vec3,
    center1: Vec3,
    t0: f32,
    t1: f32,
    radius: f32,
    radius_squared: f32,
    material: usize,
}

impl MovingSphere {
    fn new(
        center0: Vec3, center1: Vec3, t0: f32, t1: f32, radius: f32,
        material: usize,
    ) -> Self {
        MovingSphere {
            center0,
            center1,
            t0,
            t1,
            radius,
            radius_squared: radius * radius,
            material,
        }
    }

    fn center(&self, t: f32) -> Vec3 {
        self.center0
            + ((t - self.t0) / (self.t1 - self.t0))
                * (self.center1 - self.center0)
    }
}

impl Hitable for MovingSphere {
    fn hit(
        &self, r: &Ray, t_min: f32, t_max: f32, rec: &mut HitRecord,
    ) -> bool {
        let oc = r.origin - self.center(r.time);
        let a = r.direction.dot(r.direction);
        let b = oc.dot(r.direction);
        let c = oc.dot(oc) - self.radius_squared;
        let discriminat = b * b - a * c;
        if discriminat > 0.0 {
            let dsqrt = discriminat.sqrt();
            let temp = (-b - dsqrt) / a;
            if temp < t_max && temp > t_min {
                rec.t = temp;
                rec.p = r.point_at_param(temp);
                rec.normal = (rec.p - self.center(r.time)) / self.radius;
                rec.material = self.material;
                return true;
            }
            let temp = (-b + dsqrt) / a;
            if temp < t_max && temp > t_min {
                rec.t = temp;
                rec.p = r.point_at_param(temp);
                rec.normal = (rec.p - self.center(r.time)) / self.radius;
                rec.material = self.material;
                return true;
            }
        }
        return false;
    }

    fn bounding_box(&self, _: f32, _: f32) -> Option<AABB> {
        let rv = Vec3::new(self.radius, self.radius, self.radius, 0.0);
        let b0 = AABB::new(self.center0 - rv, self.center0 + rv);
        let b1 = AABB::new(self.center1 - rv, self.center1 + rv);
        Some(AABB::surround(b0, b1))
    }
}

#[derive(Debug, Clone, Copy)]
struct ConstTexture(Vec3);

impl Texture for ConstTexture {
    fn value(&self, _: f32, _: f32, _: Vec3) -> Vec3 {
        self.0
    }
}

#[derive(Debug)]
struct CheckerTexture {
    odd: Rc<dyn Texture>,
    even: Rc<dyn Texture>,
}

impl Texture for CheckerTexture {
    fn value(&self, u: f32, v: f32, p: Vec3) -> Vec3 {
        let sines = (p * 10.0).as_array();
        let o = sines[0].sin() * sines[1].sin() * sines[2].sin();
        if o < 0.0 {
            self.odd.value(u, v, p)
        } else {
            self.even.value(u, v, p)
        }
    }
}

struct Lambertian {
    albedo: Rc<dyn Texture>,
}

impl Lambertian {
    fn new(albedo: Rc<dyn Texture>) -> Self {
        Lambertian {
            albedo: albedo.clone(),
        }
    }
}

impl Material for Lambertian {
    fn scatter(
        &self, rng: &mut RNG, ray: &Ray, rec: &mut HitRecord,
        attenuation: &mut Vec3, scattered: &mut Ray,
    ) -> bool {
        let target = rec.p + rec.normal + rng.random_in_unit_sphere();
        *scattered =
            Ray::new(rec.p, target - rec.p, ray.time, ray.max_depth - 1);
        *attenuation = self.albedo.value(0.0, 0.0, rec.p);
        true
    }
}

struct Metal {
    albedo: Vec3,
    fuzz: f32,
}

#[inline(always)]
fn reflect(v: Vec3, n: Vec3) -> Vec3 {
    v - 2.0 * v.dot(n) * n
}

impl Metal {
    fn new(albedo: Vec3, fuzz: f32) -> Self {
        Metal {
            albedo,
            fuzz: if fuzz < 1.0 { fuzz } else { 1.0 },
        }
    }
}

impl Material for Metal {
    fn scatter(
        &self, rng: &mut RNG, ray: &Ray, rec: &mut HitRecord,
        attenuation: &mut Vec3, scattered: &mut Ray,
    ) -> bool {
        let reflected = reflect(ray.direction, rec.normal);
        *scattered = Ray::new(
            rec.p,
            reflected + self.fuzz * rng.random_in_unit_sphere(),
            ray.time,
            ray.max_depth - 1,
        );
        *attenuation = self.albedo;
        scattered.direction.dot(rec.normal) > 0.0
    }
}

fn refract(v: Vec3, n: Vec3, ni_over_nt: f32) -> Option<Vec3> {
    let dt = v.dot(n);
    let discriminant = 1.0 - ni_over_nt * ni_over_nt * (1.0 - dt * dt);
    if discriminant > 0.0 {
        return Some(ni_over_nt * (v - n * dt) - n * discriminant.sqrt());
    }
    None
}

struct Dielectric {
    ref_idx: f32,
}

impl Material for Dielectric {
    fn scatter(
        &self, rng: &mut RNG, ray: &Ray, rec: &mut HitRecord,
        attenuation: &mut Vec3, scattered: &mut Ray,
    ) -> bool {
        let reflected = reflect(ray.direction, rec.normal);
        let outward_normal;
        let ni_over_nt;
        let cosine;
        *attenuation = Vec3::new(1.0, 1.0, 1.0, 0.0);
        let dir_dot_nrm = ray.direction.dot(rec.normal);
        if dir_dot_nrm > 0.0 {
            outward_normal = -rec.normal;
            ni_over_nt = self.ref_idx;
            cosine = self.ref_idx * dir_dot_nrm / ray.direction.length();
        } else {
            outward_normal = rec.normal;
            ni_over_nt = 1.0 / self.ref_idx;
            cosine = -dir_dot_nrm / ray.direction.length();
        }
        let some_refracted = refract(ray.direction, outward_normal, ni_over_nt);
        let reflect_prob = if some_refracted.is_some() {
            schlick(cosine, self.ref_idx)
        } else {
            1.0
        };
        if rng.rand() < reflect_prob {
            *scattered =
                Ray::new(rec.p, reflected, ray.time, ray.max_depth - 1);
        } else {
            *scattered = Ray::new(
                rec.p,
                some_refracted.unwrap(),
                ray.time,
                ray.max_depth - 1,
            );
        }
        true
    }
}

fn schlick(cosine: f32, ref_idx: f32) -> f32 {
    let r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    let r0 = r0 * r0;
    r0 + (1.0 - r0) * (1.0 - cosine).powf(5.0)
}

struct DiffuseLight {
    emit: Rc<dyn Texture>,
}

impl Material for DiffuseLight {
    fn scatter(
        &self, rng: &mut RNG, ray: &Ray, rec: &mut HitRecord,
        attenuation: &mut Vec3, scattered: &mut Ray,
    ) -> bool {
        false
    }

    fn emitted(&self, u: f32, v: f32, p: Vec3) -> Vec3 {
        self.emit.value(u, v, p)
    }
}

fn random_scene(rng: &mut RNG, matlib: &mut MaterialLibrary) -> HitableList {
    let mut hitables = HitableList::default();

    matlib.lib.push(Box::new(Lambertian {
        albedo: Rc::new(CheckerTexture {
            odd: Rc::new(ConstTexture(Vec3::new(0.2, 0.3, 0.1, 0.0))),
            even: Rc::new(ConstTexture(Vec3::new(0.9, 0.9, 0.9, 0.0))),
        }),
    }));
    hitables.list.push(Rc::new(Sphere::new(
        Vec3::new(0.0, -1000.0, 0.0, 0.0),
        1000.0,
        matlib.lib.len() - 1,
    )));

    for a in -5..5 {
        for b in -5..5 {
            let choose_mat: f32 = rng.rand();
            let center = Vec3::new(
                a as f32 + 0.9 * rng.rand(),
                0.2,
                b as f32 + 0.9 * rng.rand(),
                0.0,
            );
            if (center - Vec3::new(4.0, 0.2, 0.0, 0.0)).length() <= 0.9 {
                continue;
            }
            if choose_mat < 0.8 {
                matlib.lib.push(Box::new(Lambertian {
                    albedo: Rc::new(ConstTexture(Vec3::new(
                        rng.rand() * rng.rand(),
                        rng.rand() * rng.rand(),
                        rng.rand() * rng.rand(),
                        0.0,
                    ))),
                }));
                hitables.list.push(Rc::new(Sphere::new(
                    center,
                    // center + Vec3::new(0.0, 0.5 * rng.rand(), 0.0),
                    // 0.0,
                    // 1.0,
                    0.2,
                    matlib.lib.len() - 1,
                )));
            } else if choose_mat < 0.95 {
                matlib.lib.push(Box::new(Metal {
                    albedo: Vec3::new(
                        0.5 * (1.0 + rng.rand()),
                        0.5 * (1.0 + rng.rand()),
                        0.5 * (1.0 + rng.rand()),
                        0.0,
                    ),
                    fuzz: 0.5 * rng.rand(),
                }));
                hitables.list.push(Rc::new(Sphere::new(
                    center,
                    0.2,
                    matlib.lib.len() - 1,
                )));
            } else {
                matlib.lib.push(Box::new(Dielectric { ref_idx: 1.5 }));
                hitables.list.push(Rc::new(Sphere::new(
                    center,
                    0.2,
                    matlib.lib.len() - 1,
                )));
            }
        }
    }

    matlib.lib.push(Box::new(Dielectric { ref_idx: 1.5 }));
    hitables.list.push(Rc::new(Sphere::new(
        Vec3::new(0.0, 1.0, 0.0, 0.0),
        1.0,
        matlib.lib.len() - 1,
    )));

    // matlib.lib.push(Box::new(Lambertian {
    //     albedo: Rc::new(ConstTexture(Vec3::new(0.4, 0.2, 0.1))),
    // }));
    matlib.lib.push(Box::new(DiffuseLight {
        emit: Rc::new(ConstTexture(Vec3::new(4.0, 4.0, 4.0, 0.0))),
    }));
    hitables.list.push(Rc::new(Sphere::new(
        Vec3::new(4.0, 1.0, 0.0, 0.0),
        1.0,
        matlib.lib.len() - 1,
    )));

    matlib.lib.push(Box::new(Metal {
        albedo: Vec3::new(0.7, 0.6, 0.5, 0.0),
        fuzz: 0.0,
    }));
    hitables.list.push(Rc::new(Sphere::new(
        Vec3::new(-4.0, 1.0, 0.0, 0.0),
        1.0,
        matlib.lib.len() - 1,
    )));

    hitables
}

struct Progress {
    current: usize,
    total: usize,
}

impl Progress {
    fn new(total: usize) -> Self {
        Progress { current: 0, total }
    }

    fn increment(&mut self) {
        self.current += 1
    }

    fn update(&mut self, current: usize) {
        self.current = current
    }

    fn fraction(&self) -> f32 {
        self.current as f32 / self.total as f32
    }
}

fn main() {
    let width: usize = 400;
    let height: usize = 200;
    let samples: usize = 100;
    let ray_depth: usize = 50;

    let mut rng = RNG::default();

    let mut matlib = MaterialLibrary::default();
    let scene = random_scene(&mut rng, &mut matlib);
    let bvh = BVH::new(scene, 0.0, 0.1, &mut rng);
    //::std::dbg!(&bvh);

    let look_from = Vec3::new(13.0, 2.0, 3.0, 0.0);
    let look_at = Vec3::new(0.0, 0.0, 0.0, 0.0);
    let dist_to_focus = 10.0; //(look_from - look_at).length();
    let aperture = 0.0;
    let fov = 20.0;
    let cam = Camera::new(
        look_from,
        look_at,
        &Vec3::new(0.0, 1.0, 0.0, 0.0),
        fov,
        (width as f32) / (height as f32),
        aperture,
        dist_to_focus,
        0.0,
        1.0,
    );

    let _ = io::stderr().write_all(b"Setup complete\n");

    let mut progress = Progress::new(width * height);

    print!("P3\n{} {}\n255\n", width, height);
    for y in (0..height).rev() {
        for x in 0..width {
            let mut color = Vec3::new(0.0, 0.0, 0.0, 0.0);
            for _ in 0..samples {
                let u = (x as f32 + rng.rand()) / (width as f32);
                let v = (y as f32 + rng.rand()) / (height as f32);
                let r = cam.get_ray(&mut rng, u, v, ray_depth);
                color += r.trace(&mut rng, &bvh, &matlib, 0);
            }
            color /= samples as f32;
            color = color.sqrt(); // Vec3::new(color.r.sqrt(), color.g.sqrt(), color.b.sqrt());
            let (r, g, b, _) = color.as_u8();
            print!("{} {} {}\n", r, g, b);
            progress.increment();
        }
    }
}
