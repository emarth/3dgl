use image::{GrayImage, Luma};

// Point and Vector
#[derive(Debug, Clone)]
pub struct Point {
    x: f32,
    y: f32,
    z: f32,
}

impl Point {
    fn plus(&self, other : &Vector) -> Point{
        Point {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Vector {
    x: f32,
    y: f32,
    z: f32,
}

pub fn from_points(p: &Point, q : &Point) -> Vector {
    Vector { x: q.x - p.x, y: q.y - p.y, z: q.z - p.z }
}

impl Vector {

    pub fn scale_by(&mut self, c : f32) {
        self.x = self.x*c;
        self.y = self.y*c;
        self.z = self.z*c;
    }

    pub fn scaled_by(&self, c: f32) -> Vector{
        let mut scaled = self.clone();
        scaled.scale_by(c);
        scaled
    }

    fn norm(&self) -> f32 {
        (self.x*self.x + self.y*self.y + self.z*self.z).sqrt()
    }

    pub fn normalize(&mut self) {
        let inv_norm = 1.0/self.norm();
        self.scale_by(inv_norm);
    }

    pub fn add_to(&mut self, other : &Vector) {
        self.x = self.x + other.x;
        self.y = self.y + other.y;
        self.z = self.z + other.z;
    }

    pub fn dot(&self, other: &Vector) -> f32 {
        self.x*other.x + self.y*other.y + self.z*other.z
    }

    pub fn cross(&self, other: &Vector) -> Vector {
        Vector {
            x: self.y*other.z - self.z*other.y,
            y: self.z*other.x - self.x*other.z,
            z: self.x*other.y - self.y*other.x,
        }
    }

    pub fn reflect_across(&mut self, other : &Vector) { // adds scaling but we normalize after anyways, assumes other has magnitude 1
        let proj = other.scaled_by(self.dot(other));
        self.add_to(&proj.scaled_by(-1.0));
        self.scale_by(-1.0);
        self.add_to(&proj);
    }

}

// Projections

pub struct Projection {
    focal : Point,
    plane_centre : Point,
    width_unit : Vector,
    height_unit : Vector,
    width: u32,
    height: u32,
}

pub fn new_projection(f: Point, p: Point, i: Vector, j: Vector, w: u32, h: u32) -> Result<Projection, ()> {
    let dot = i.dot(&j);
    if dot == 0.0 {
        Ok(Projection {
            focal: f, plane_centre: p, width_unit: i, height_unit: j, width: w, height: h
        })
    } else {
    Err(())
    }
}

pub struct HalfLine {
    p : Point,
    direction : Vector,
}

pub trait Renderable {
    fn intersection(&self, line_of_sight : &HalfLine) -> Option<Point>; //none if half-line does not intersect self
    fn normal(&self, p: &Point) -> Option<Vector>; // None if point is not in self
}

impl Projection {

    fn render_pixel(&self, objects: &Vec<&dyn Renderable>, lights: &Vec<Point>, ambient: f32, w: u32, h: u32) -> Luma<u8> {
        let x = w as f32 - self.width as f32 / 2.0;
        let y = self.height as f32 / 2.0 - h as f32;

        let mut offset = self.width_unit.scaled_by(x);
        offset.add_to(&self.height_unit.scaled_by(y));

        let los = HalfLine {
            p: self.focal.clone(),
            direction: from_points(&self.focal, &self.plane_centre.plus(&offset)),
        };
        let intersections : Vec<Option<Point>> = objects.iter().map(|obj| {
            obj.intersection(&los)
        }).collect();
        // find the closest point
        let min_index = { 
            let mut current = None;
            let mut current_value: f32 = f32::MAX;
            for i in 0..intersections.len() {
                match &intersections[i] {
                    None => {},
                    Some(pt) => {
                        let candidate = from_points(&self.focal, pt).norm();
                        if  candidate < current_value {
                            current = Some(i);
                            current_value = candidate;
                        }
                    }
                }
            }
            current
        };
        match min_index {
            None => image::Luma::<u8>::from([0]),
            // calculate brightness
            Some(i) => {
                let intersection = intersections[i].as_ref().unwrap();
                let normal_vector = objects[i].normal(intersection).unwrap();
                let mut brightness = ambient*(normal_vector.dot(&los.direction)/los.direction.norm()).abs();
                'lights: for light in lights {
                    let mut light_vector = from_points(intersection, light);
                    // check for objects casting a shadow
                    for obj in objects {
                        match obj.intersection(&HalfLine {
                            p: intersection.plus(&light_vector.scaled_by(0.01)),
                            direction: light_vector.clone()
                        }) {
                            None => (),
                            Some(_) => {continue 'lights;}
                        }
                    }

                    light_vector.normalize();
                    light_vector.reflect_across(&normal_vector);
                    let dot = -light_vector.dot(&los.direction)/los.direction.norm();
                    if dot > 0.0 {brightness += (1.0-brightness)*dot*dot};
                }
                image::Luma::<u8>::from([(brightness*255.0) as u8])
            }
        }
    }

    pub fn render(&self, objects: &Vec<&dyn Renderable>, lights: &Vec<Point>, ambient: f32) -> GrayImage {
        // implement parallel later
        GrayImage::from_fn(self.width, self.height, |w,h| {self.render_pixel(objects, lights, ambient, w, h)})
    }
}

// Example with a sphere

pub struct Sphere {
    centre: Point,
    radius: f32,
}

impl Renderable for Sphere {
    fn intersection(&self, line_of_sight : &HalfLine) -> Option<Point> {
        let v = &line_of_sight.direction;
        let o = &self.centre;
        let p = &line_of_sight.p;
        let degree_two = v.x*v.x + v.y*v.y + v.z*v.z;
        let degree_one = 2.0*(v.x*(p.x - o.x) + v.y*(p.y - o.y) + v.z*(p.z - o.z));
        let degree_zero = (p.x - o.x)*(p.x - o.x) + (p.y - o.y)*(p.y - o.y) + (p.z - o.z)*(p.z - o.z) - self.radius*self.radius;

        let d = degree_one*degree_one - 4.0*degree_two*degree_zero;
        let t;

        if d < 0.0 {return None}
        else {
            let t_neg = -0.5/degree_two*(degree_one + d.sqrt()); 
            let t_pos = -0.5/degree_two*(degree_one - d.sqrt()); 
            if t_neg > 0.0 {t = t_neg;}
            else if t_pos > 0.0 {t = t_pos;}
            else {return None}
            Some(p.plus(&v.scaled_by(t)))
        }
    }

    fn normal(&self, p: &Point) -> Option<Vector> {

        let mut v = from_points(&self.centre, p);
        if (v.norm() - self.radius).abs() < 0.001 { //leeway for floating point stuff
            v.normalize();
            return Some(v);
        }
        None
    }
}

// Example with triangle

struct Triangle {
    points: [Point; 3],
}

fn solve(matrix : &mut [Vector; 3], p: &Vector) -> Result<Vector, ()>{

    // Worst Jordan elimination routine ever written
    let mut q = p.clone();
    let mut c : f32;

    let mut top : Option<usize> = None;
    for i in 0..3 {
        if matrix[i].x != 0.0 {top = Some(i); break}
    }
    match top {
        None => return Err(()),
        Some(n) => {
            if n > 0 {
                let temp = matrix[0].clone();
                matrix[0] = matrix[n].clone();
                matrix[n] = temp;

                let tq = q.x;
                match n {
                    1 => {q.x = q.y; q.y = tq},
                    2 => {q.x = q.z; q.z = tq},
                    _ => ()
                }
            }
        }
    }
    c = -matrix[1].x/matrix[0].x;
    matrix[1].add_to(&(matrix[0].scaled_by(c)));
    q.y += c*q.x;
    c = -matrix[2].x/matrix[0].x;
    matrix[2].add_to(&(matrix[0].scaled_by(c)));
    q.z += c*q.x;


    top = None;
    for i in 1..3 {
        if matrix[i].y != 0.0 {top = Some(i); break}
    }
    match top {
        None => return Err(()),
        Some(n) => {
            if n > 1 {
                let temp = matrix[1].clone();
                matrix[1] = matrix[n].clone();
                matrix[n] = temp;
                let tq = q.y;
                q.y = q.z; q.z = tq;
            }
        }
    }

    if matrix[2].z == 0.0 {return Err(())}

    c = -matrix[2].y/matrix[1].y;
    matrix[2].add_to(&matrix[1].scaled_by(c));
    q.z += c*q.y;

    q.y -= q.z*matrix[1].z/matrix[2].z;
    q.x -= q.z*matrix[0].z/matrix[2].z;
    q.x -= q.y*matrix[0].y/matrix[1].y;

    q.x /= matrix[0].x;
    q.y /= matrix[1].y;
    q.z /= matrix[2].z;

    Ok(q)
}

impl Renderable for Triangle {
    fn intersection(&self, line_of_sight : &HalfLine) -> Option<Point> {
        let v = from_points(&self.points[0], &self.points[1]);
        let w = from_points(&self.points[0], &self.points[2]);
        let mut matrix = [
            Vector {x: v.x, y: w.x, z: line_of_sight.direction.x},
            Vector {x: v.y, y: w.y, z: line_of_sight.direction.y},
            Vector {x: v.z, y: w.z, z: line_of_sight.direction.z},
        ];

        let solution = solve(&mut matrix, &from_points(&self.points[0], &line_of_sight.p));
        match solution {
            Ok(v) => if v.x >= 0.0 && v.y >= 0.0 && v.x + v.y < 1.0 && v.z <= 0.0 {
                return Some(line_of_sight.p.plus(&line_of_sight.direction.scaled_by(-v.z)))
            } else {return None},
            Err(()) => {return None }
        }
    }

    fn normal(&self, _: &Point) -> Option<Vector> {
        let v = from_points(&self.points[0], &self.points[1]);
        let w = from_points(&self.points[0], &self.points[2]);
        let mut n = v.cross(&w);
        n.normalize();
        Some(n)
    }
}

struct Rectangle{
    points: [Point; 3],
}

impl Renderable for Rectangle {
    fn intersection(&self, line_of_sight : &HalfLine) -> Option<Point> {
        let v = from_points(&self.points[0], &self.points[1]);
        let w = from_points(&self.points[0], &self.points[2]);
        let mut matrix = [
            Vector {x: v.x, y: w.x, z: line_of_sight.direction.x},
            Vector {x: v.y, y: w.y, z: line_of_sight.direction.y},
            Vector {x: v.z, y: w.z, z: line_of_sight.direction.z},
        ];

        let solution = solve(&mut matrix, &from_points(&self.points[0], &line_of_sight.p));
        match solution {
            Ok(v) => if v.x >= 0.0 && v.y >= 0.0 && v.x < 1.0 && v.y < 1.0 && v.z <= 0.0 {
                return Some(line_of_sight.p.plus(&line_of_sight.direction.scaled_by(-v.z)))
            } else {return None},
            Err(()) => {return None }
        }
    }

    fn normal(&self, _: &Point) -> Option<Vector> {
        let v = from_points(&self.points[0], &self.points[1]);
        let w = from_points(&self.points[0], &self.points[2]);
        let mut n = v.cross(&w);
        n.normalize();
        Some(n)
    }
}

#[cfg(test)]
mod tests {
    use image::codecs::png::PngEncoder;
    use std::fs::File;

    use super::*;

    #[test]
    fn draw_pyramid() { 
        let vertices = vec![
            Point {x: 0.0, y: 0.0, z: 0.0},
            Point {x: 2.0, y: 3.0, z: 0.0},
            Point {x: 4.0, y:0.0, z: 4.0},
            Point {x: 4.0, y: 0.0, z: -4.0}
        ];

        let tetrahedron: Vec<Triangle> = (0..4).map(|n| {
            Triangle {
                points: [vertices[n%4].clone(), vertices[(n+1)%4].clone(), vertices[(n+2)%4].clone()]
            }
        }).collect();

        let mut objects : Vec<&dyn Renderable> = vec![];

        for face in tetrahedron.iter() {objects.push(face)}

        let proj = Projection {
            focal: Point {x: -10.0, y:10.0, z:10.0},
            plane_centre: Point {x: -5.0, y: 6.0, z: 8.0},
            width_unit: Vector {x:0.0, y: 0.0, z: 0.005},
            height_unit: Vector {x: 0.0, y: 0.005, z: 0.0},
            width: 1200,
            height: 800,
        };

        let light3 = Point {x: -50.0, y: 0.0, z: 100.0};
        let light4 = Point {x: 0.0, y: 100.0, z: 0.0};

        let buffer = File::create("tet.png").unwrap();
        let image = proj.render(&objects, &vec![light3, light4], 0.33);
        let enc = PngEncoder::new(buffer);
        image.write_with_encoder(enc).unwrap();
    }

    #[test]
    fn draw_spheres() {
        let s = Sphere {
            centre: Point {x:0.0, y:0.0, z:0.0,},
            radius: 3.0,
        };

        let s2 = Sphere {
            centre: Point {x: -4.0, y: 0.0, z: 2.0},
            radius: 0.5,
        };

        let plane = Rectangle {
            points: [
                Point {x: -50.0, y: -10.0, z: -50.0},
                Point {x: -50.0, y: -10.0, z: 50.0},
                Point {x: 50.0, y: -10.0, z: -50.0},
            ]
        };

        let proj = Projection {
            focal: Point {x: -10.0, y:0.0, z:0.0},
            plane_centre: Point {x: -5.0, y: 0.0, z: 0.0},
            width_unit: Vector {x:0.0, y: 0.0, z: 0.005},
            height_unit: Vector {x: 0.0, y: 0.005, z: 0.0},
            width: 1200,
            height: 800,
        };

        let light3 = Point {x: -50.0, y: 0.0, z: 100.0};
        let light4 = Point {x: 0.0, y: 100.0, z: 0.0};

        let objects : Vec<&dyn Renderable> = vec![&s, &s2, &plane];
        let buffer = File::create("orb.png").unwrap();
        let image = proj.render(&objects, &vec![light3, light4], 0.33);
        let enc = PngEncoder::new(buffer);
        image.write_with_encoder(enc).unwrap();
    }

}
