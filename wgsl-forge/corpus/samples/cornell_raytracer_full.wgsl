const pi = 3.14159265359;

// Quad describes 2D rectangle on a plane
struct Quad {
  // The surface plane
  plane    : vec4f,
  // A plane with a normal in the 'u' direction, intersecting the origin, at
  // right-angles to the surface plane.
  // The dot product of 'right' with a 'vec4(pos, 1)' will range between [-1..1]
  // if the projected point is within the quad.
  right    : vec4f,
  // A plane with a normal in the 'v' direction, intersecting the origin, at
  // right-angles to the surface plane.
  // The dot product of 'up' with a 'vec4(pos, 1)' will range between [-1..1]
  // if the projected point is within the quad.
  up       : vec4f,
  // The diffuse color of the quad
  color    : vec3f,
  // Emissive value. 0=no emissive, 1=full emissive.
  emissive : f32,
};

// Ray is a start point and direction.
struct Ray {
  start : vec3f,
  dir   : vec3f,
}

// Value for HitInfo.quad if no intersection occured.
const kNoHit = 0xffffffff;

// HitInfo describes the hit location of a ray-quad intersection
struct HitInfo {
  // Distance along the ray to the intersection
  dist : f32,
  // The quad index that was hit
  quad : u32,
  // The position of the intersection
  pos : vec3f,
  // The UVs of the quad at the point of intersection
  uv : vec2f,
}

// CommonUniforms uniform buffer data
struct CommonUniforms {
  // Model View Projection matrix
  mvp : mat4x4f,
  // Inverse of mvp
  inv_mvp : mat4x4f,
  // Random seed for the workgroup
  seed : vec3u,
}

// The common uniform buffer binding.
@group(0) @binding(0) var<uniform> common_uniforms : CommonUniforms;

// The quad buffer binding.
@group(0) @binding(1) var<storage> quads : array<Quad>;

// intersect_ray_quad will check to see if the ray 'r' intersects the quad 'q'.
// If an intersection occurs, and the intersection is closer than 'closest' then
// the intersection information is returned, otherwise 'closest' is returned.
fn intersect_ray_quad(r : Ray, quad : u32, closest : HitInfo) -> HitInfo {
  let q = quads[quad];
  let plane_dist = dot(q.plane, vec4(r.start, 1));
  let ray_dist = plane_dist / -dot(q.plane.xyz, r.dir);
  let pos = r.start + r.dir * ray_dist;
  let uv = vec2(dot(vec4f(pos, 1), q.right),
                dot(vec4f(pos, 1), q.up)) * 0.5 + 0.5;
  let hit = plane_dist > 0 &&
            ray_dist > 0 &&
            ray_dist < closest.dist &&
            all((uv > vec2f()) & (uv < vec2f(1)));
  return HitInfo(
    select(closest.dist, ray_dist, hit),
    select(closest.quad, quad,     hit),
    select(closest.pos,  pos,      hit),
    select(closest.uv,   uv,       hit),
  );
}

// raytrace finds the closest intersecting quad for the given ray
fn raytrace(ray : Ray) -> HitInfo {
  var hit = HitInfo();
  hit.dist = 1e20;
  hit.quad = kNoHit;
  for (var quad = 0u; quad < arrayLength(&quads); quad++) {
    hit = intersect_ray_quad(ray, quad, hit);
  }
  return hit;
}

// A pseudo random number. Initialized with init_rand(), updated with rand().
var<private> rnd : vec3u;

// Initializes the random number generator.
fn init_rand(invocation_id : vec3u) {
  const A = vec3(1741651 * 1009,
                 140893  * 1609 * 13,
                 6521    * 983  * 7 * 2);
  rnd = (invocation_id * A) ^ common_uniforms.seed;
}

// Returns a random number between 0 and 1.
fn rand() -> f32 {
  const C = vec3(60493  * 9377,
                 11279  * 2539 * 23,
                 7919   * 631  * 5 * 3);

  rnd = (rnd * C) ^ (rnd.yzx >> vec3(4u));
  return f32(rnd.x ^ rnd.y) / f32(0xffffffff);
}

// Returns a random point within a unit sphere centered at (0,0,0).
fn rand_unit_sphere() -> vec3f {
    var u = rand();
    var v = rand();
    var theta = u * 2.0 * pi;
    var phi = acos(2.0 * v - 1.0);
    var r = pow(rand(), 1.0/3.0);
    var sin_theta = sin(theta);
    var cos_theta = cos(theta);
    var sin_phi = sin(phi);
    var cos_phi = cos(phi);
    var x = r * sin_phi * sin_theta;
    var y = r * sin_phi * cos_theta;
    var z = r * cos_phi;
    return vec3f(x, y, z);
}

fn rand_concentric_disk() -> vec2f {
    let u = vec2f(rand(), rand());
    let uOffset = 2.f * u - vec2f(1, 1);

    if (uOffset.x == 0 && uOffset.y == 0){
        return vec2f(0, 0);
    }

    var theta = 0.0;
    var r = 0.0;
    if (abs(uOffset.x) > abs(uOffset.y)) {
        r = uOffset.x;
        theta = (pi / 4) * (uOffset.y / uOffset.x);
    } else {
        r = uOffset.y;
        theta = (pi / 2) - (pi / 4) * (uOffset.x / uOffset.y);
    }
    return r * vec2f(cos(theta), sin(theta));
}

fn rand_cosine_weighted_hemisphere() -> vec3f {
    let d = rand_concentric_disk();
    let z = sqrt(max(0.0, 1.0 - d.x * d.x - d.y * d.y));
    return vec3f(d.x, d.y, z);
}
// The lightmap data
@group(1) @binding(0) var lightmap : texture_2d_array<f32>;

// The sampler used to sample the lightmap
@group(1) @binding(1) var smpl : sampler;

// The output framebuffer
@group(1) @binding(2) var framebuffer : texture_storage_2d<rgba16float, write>;

override WorkgroupSizeX : u32;
override WorkgroupSizeY : u32;

const NumReflectionRays = 5;

@compute @workgroup_size(WorkgroupSizeX, WorkgroupSizeY)
fn main(@builtin(global_invocation_id) invocation_id : vec3u) {
  if (all(invocation_id.xy < textureDimensions(framebuffer))) {
    init_rand(invocation_id);

    // Calculate the fragment's NDC coordinates for the intersection of the near
    // clip plane and far clip plane
    let uv = vec2f(invocation_id.xy) / vec2f(textureDimensions(framebuffer).xy);
    let ndcXY = (uv - 0.5) * vec2(2, -2);

    // Transform the coordinates back into world space
    var near = common_uniforms.inv_mvp * vec4f(ndcXY, 0.0, 1);
    var far = common_uniforms.inv_mvp * vec4f(ndcXY, 1, 1);
    near /= near.w;
    far /= far.w;

    // Create a ray that starts at the near clip plane, heading in the fragment's
    // z-direction, and raytrace to find the nearest quad that the ray intersects.
    let ray = Ray(near.xyz, normalize(far.xyz - near.xyz));
    let hit = raytrace(ray);

    let hit_color = sample_hit(hit);
    var normal = quads[hit.quad].plane.xyz;

    // Fire a few rays off the surface to collect some reflections
    let bounce = reflect(ray.dir, normal);
    var reflection : vec3f;
    for (var i = 0; i < NumReflectionRays; i++) {
      let reflection_dir = normalize(bounce + rand_unit_sphere()*0.1);
      let reflection_ray = Ray(hit.pos + bounce * 1e-5, reflection_dir);
      let reflection_hit = raytrace(reflection_ray);
      reflection += sample_hit(reflection_hit);
    }
    let color = mix(reflection / NumReflectionRays, hit_color, 0.95);

    textureStore(framebuffer, invocation_id.xy, vec4(color, 1));
  }
}


// Returns the sampled hit quad's lightmap at 'hit.uv', and adds the quad's
// emissive value.
fn sample_hit(hit : HitInfo) -> vec3f {
  let quad = quads[hit.quad];
  // Sample the quad's lightmap, and add emissive.
  return textureSampleLevel(lightmap, smpl, hit.uv, hit.quad, 0).rgb +
         quad.emissive * quad.color;
}
