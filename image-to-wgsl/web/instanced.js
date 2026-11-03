// Instanced renderer for image-to-wgsl splat lists.
//
// The per-pixel shader (trainer.js emitWGSL, codec.py emit_wgsl) loops over every splat at every pixel:
// cost = pixels x splats. Here each splat is one quad, sized to where its weight falls below 1/1024, so
// the fragment shader only runs where the splat has ink, and the GPU's blending does the painting:
// "over" blending, src * a + dst * (1 - a), is exactly mix(c, colour, w). The splat list order is the
// painting order, so no sorting is needed, and drawing the first n instances paints the n-splat prefix.
// Blending happens in a float target (8-bit blending would round at every layer): rgba32float where the
// device has float32-blendable, else rgba16float. A last pass copies it to the screen. Same packed data, same decode arithmetic as the per-pixel shader.
//
// Entry points: vb/fb background, vs/fs splats (instanced, 6 vertices each), vc/fc copy.
// Bindings: @group(0) @binding(0) uniform {a: vec4f(width, height, zoom, A), b: vec4f(x, y, sx, 0)}, where (x, y)
// is the image point (as a fraction of its width and height) at the centre of the view and sx stretches x (1 on
// screen; a deep-zoom crop of a resampled level sets it so the crop maps exactly); @binding(1) the
// float texture (copy). Zoom 1 with (0.5, 0.5) fits the whole image. Splats are redrawn at every zoom, so
// edges stay smooth at any magnification.
import { LAYOUT, packBits } from "./trainer.js";

const B = LAYOUT.bits, MAX = B.map((b) => 2 ** b - 1), MBG = 2 ** LAYOUT.bbg - 1;
const SPAN = (LAYOUT.sHi - LAYOUT.sLo) / MAX[2];
const CUTOFF = 1024;   // a splat's quad ends where alpha x falloff < 1/CUTOFF

function fnum(v) {
  let s = String(parseFloat(Math.fround(v).toPrecision(9)));
  if (!/[.e]/.test(s)) s += ".";
  return s;
}

export function emitInstanced(r, n = null) {
  const N = n ?? r.n, A = r.A;
  const fields = [];
  for (const v of r.bg) fields.push([v, LAYOUT.bbg]);
  for (let i = 0; i < N; i++) for (let f = 0; f < 10; f++) fields.push([r.P[i * 10 + f], B[f]]);
  let words = packBits(fields);
  // WebKit takes const arrays of at most 2047 elements. Past 2047 words, four words go in each vec4u
  // element; past 2047 of those (about 4,295 splats), the vec4u elements go in several arrays.
  let data;
  if (words.length <= 2047) data = `const D=array<u32,${words.length}>(${Array.from(words).join(",")});fn dw(i:u32)->u32{return D[i];}`;
  else {
    const pad = new Uint32Array(Math.ceil(words.length / 4) * 4); pad.set(words); words = pad;
    const v = Array.from({ length: words.length / 4 }, (_, i) => `vec4u(${words.slice(4 * i, 4 * i + 4).join(",")})`), parts = [];
    for (let i = 0; i < v.length; i += 2047) parts.push(v.slice(i, i + 2047));
    data = parts.map((a, k) => `const D${k}=array<vec4u,${a.length}>(${a.join(",")});`).join("") +
      `fn dw(i:u32)->u32{let j=i>>2u;let c=i&3u;` + parts.slice(0, -1).map((_, k) => `if(j<${2047 * (k + 1)}u){return D${k}[j-${2047 * k}u][c];}`).join("") +
      `return D${parts.length - 1}[j-${2047 * (parts.length - 1)}u][c];}`;
  }
  const off = [0]; for (const b of B) off.push(off[off.length - 1] + b);
  const fld = (j) => `r(o+${off[j]}u,${B[j]}u)`;
  const bgv = (j) => `vec3f(r(${3 * j * 6}u,6u),r(${(3 * j + 1) * 6}u,6u),r(${(3 * j + 2) * 6}u,6u))`;
  const sig = (j) => `exp2(${fnum(LAYOUT.sLo)}+${fld(j)}*${fnum(SPAN)})`;
  return `struct U{a:vec4f,b:vec4f}@group(0)@binding(0)var<uniform>u:U;@group(0)@binding(1)var tex:texture_2d<f32>;const A=${fnum(A)};const N=${N}u;` + data +
    `fn r(o:u32,n:u32)->f32{let w=o>>5u;let s=o&31u;var v=dw(w)>>s;if(s+n>32u){v|=dw(w+1u)<<(32u-s);}return f32(v&((1u<<n)-1u));}` +
    // image placement: q (height units) = (p - off) / k, with k pixels per height unit after zoom
    `fn kk()->vec2f{let s=min(u.a.x/A,u.a.y)*u.a.z;return vec2f(s*u.b.z,s);}fn oo()->vec2f{return u.a.xy*.5-vec2f(u.b.x*A,u.b.y)*kk();}` +
    `fn inside(p:vec2f)->bool{let q=(p-oo())/kk();return!(q.x<0.||q.y<0.||q.x>A||q.y>1.);}` +
    `fn tri(i:u32)->vec4f{return vec4f(vec2f(f32(i%2u),f32(i/2u))*4.-1.,0.,1.);}` +
    `@vertex fn vb(@builtin(vertex_index)i:u32)->@builtin(position)vec4f{return tri(i);}` +
    `@fragment fn fb(@builtin(position)p:vec4f)->@location(0)vec4f{if(!inside(p.xy)){return vec4f(1.);}let h=((p.x-oo().x)/kk().x)/A;let y=(p.y-oo().y)/kk().y;` +
    `return vec4f(mix(mix(${bgv(0)},${bgv(1)},h),mix(${bgv(2)},${bgv(3)},h),y)/${fnum(MBG)},1.);}` +
    `struct V{@builtin(position)pos:vec4f,@location(0)l:vec2f,@location(1)@interpolate(flat)c:vec4f,@location(2)@interpolate(flat)m:u32}` +
    `@vertex fn vs(@builtin(vertex_index)vi:u32,@builtin(instance_index)ii:u32)->V{let o=${12 * LAYOUT.bbg}u+ii*${B.reduce((a, b) => a + b, 0)}u;` +
    `let ctr=vec2f(${fld(0)}*${fnum(A / MAX[0])},${fld(1)}*${fnum(1 / MAX[1])});let sx=${sig(2)};let sy=${sig(3)};let t=${fld(4)}*${fnum(Math.PI / 2 ** B[4])};` +
    `let cs=cos(t);let sn=sin(t);let m=u32(${fld(9)});let al=(${fld(8)}+1.)*${fnum(1 / 2 ** B[8])};` +
    // radius in sigma units where al * exp(-.5 * d2^(2^m)) = 1/CUTOFF
    `let rr=sqrt(pow(max(2.*log(al*${fnum(CUTOFF)}),1e-3),1./f32(1u<<m)));` +
    `var cn=array<vec2f,6>(vec2f(-1.,-1.),vec2f(1.,-1.),vec2f(-1.,1.),vec2f(-1.,1.),vec2f(1.,-1.),vec2f(1.,1.));let l=cn[vi]*rr;` +
    `let a=l.x*sx;let b=l.y*sy;let q=ctr+vec2f(a*cs-b*sn,a*sn+b*cs);let p=q*kk()+oo();` +
    `var v:V;v.pos=vec4f(p.x/u.a.x*2.-1.,1.-p.y/u.a.y*2.,0.,1.);v.l=l;` +
    `v.c=vec4f(vec3f(${fld(5)},${fld(6)},${fld(7)})*${fnum(1 / MAX[5])},al);v.m=m;return v;}` +
    `@fragment fn fs(v:V)->@location(0)vec4f{if(!inside(v.pos.xy)){discard;}var e=min(dot(v.l,v.l),40.);for(var j=0u;j<v.m;j++){e*=e;}` +
    `return vec4f(v.c.rgb,v.c.a*exp(-.5*e));}` +
    `@vertex fn vc(@builtin(vertex_index)i:u32)->@builtin(position)vec4f{return tri(i);}` +
    `@fragment fn fc(@builtin(position)p:vec4f)->@location(0)vec4f{return vec4f(textureLoad(tex,vec2i(p.xy),0).rgb,1.);}`;
}

// Host side. draw(view, w, h, n, cam, nd) paints the first n splats into `view` (w x h pixels, format `fmt`), then
// the first nd deep-zoom detail splats; cam = {z, x, y, sx} zooms by z about image point (x, y) (fractions of its
// width and height), default fit. Pass a device created with the "float32-blendable" feature, when the adapter has
// it, for exact blending. Kept free of imports so a downloaded page can embed it with createRenderer.toString().
//
// Detail splats (setDetail) come from deep-zoom tiles, each a small splat list trained on a crop of a
// higher-resolution level over what the coarser levels paint there. They are decoded on the host into image units
// (the base list's: x in [0, A], y in [0, 1]), 12 floats each: centre, sigmas; cos, sin, shape exponent, alpha;
// colour, tile. Each tile has a clip rectangle (4 floats, image units), so a tile paints only its own core and the
// margin it was trained on stays hidden. Tiles are stored coarse level first, so drawing a prefix of nd splats
// draws every tile up to a level. readback() returns the float target, which the deep zoom trains against.
export function createRenderer(dev, fmt, code) {
  const A = +/const A=([^;]+);/.exec(code)[1];
  const mod = dev.createShaderModule({ code });
  const ub = dev.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const OVER = { color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
                 alpha: { srcFactor: "zero", dstFactor: "one", operation: "add" } };
  const mk = (m, v, f, format, blend) => dev.createRenderPipeline({ layout: "auto", vertex: { module: m, entryPoint: v },
    fragment: { module: m, entryPoint: f, targets: [{ format, blend }] }, primitive: { topology: "triangle-list" } });
  const FT = dev.features.has("float32-blendable") ? "rgba32float" : "rgba16float";
  const pBg = mk(mod, "vb", "fb", FT), pSp = mk(mod, "vs", "fs", FT, OVER), pCp = mk(mod, "vc", "fc", fmt);
  const ubg = (p) => dev.createBindGroup({ layout: p.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: ub } }] });
  const gBg = ubg(pBg), gSp = ubg(pSp);
  let tex = null, gCp = null, tw = 0, th = 0, pDt = null, gDt = null, dBufs = [], nDet = 0;
  const DETAIL = `struct U{a:vec4f,b:vec4f}@group(0)@binding(0)var<uniform>u:U;@group(0)@binding(2)var<storage,read>S:array<vec4f>;@group(0)@binding(3)var<storage,read>R:array<vec4f>;
fn kk()->vec2f{let s=min(u.a.x/u.a.w,u.a.y)*u.a.z;return vec2f(s*u.b.z,s);}fn oo()->vec2f{return u.a.xy*.5-vec2f(u.b.x*u.a.w,u.b.y)*kk();}
struct V{@builtin(position)pos:vec4f,@location(0)l:vec2f,@location(1)@interpolate(flat)c:vec4f,@location(2)@interpolate(flat)m:u32,@location(3)q:vec2f,@location(4)@interpolate(flat)r:vec4f}
@vertex fn vd(@builtin(vertex_index)vi:u32,@builtin(instance_index)ii:u32)->V{let v0=S[ii*3u];let v1=S[ii*3u+1u];let v2=S[ii*3u+2u];let m=u32(v1.z);
let rr=sqrt(pow(max(2.*log(v1.w*1024.),1e-3),1./f32(1u<<m)));var cn=array<vec2f,6>(vec2f(-1.,-1.),vec2f(1.,-1.),vec2f(-1.,1.),vec2f(-1.,1.),vec2f(1.,-1.),vec2f(1.,1.));
let l=cn[vi]*rr;let a=l.x*v0.z;let b=l.y*v0.w;let q=v0.xy+vec2f(a*v1.x-b*v1.y,a*v1.y+b*v1.x);let p=q*kk()+oo();
var v:V;v.pos=vec4f(p.x/u.a.x*2.-1.,1.-p.y/u.a.y*2.,0.,1.);v.l=l;v.c=vec4f(v2.rgb,v1.w);v.m=m;v.q=q;v.r=R[u32(v2.w)];return v;}
@fragment fn fd(v:V)->@location(0)vec4f{if(v.q.x<v.r.x||v.q.y<v.r.y||v.q.x>=v.r.z||v.q.y>=v.r.w){discard;}var e=min(dot(v.l,v.l),40.);for(var j=0u;j<v.m;j++){e*=e;}
return vec4f(v.c.rgb,v.c.a*exp(-.5*e));}`;
  function target(w, h) {
    if (w === tw && h === th) return;
    tex?.destroy();
    tex = dev.createTexture({ size: [w, h], format: FT, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
    gCp = dev.createBindGroup({ layout: pCp.getBindGroupLayout(0), entries: [{ binding: 1, resource: tex.createView() }] });
    tw = w; th = h;
  }
  function paint(enc, w, h, n, cam, nd) {
    target(w, h);
    dev.queue.writeBuffer(ub, 0, new Float32Array([w, h, cam?.z ?? 1, A, cam?.x ?? 0.5, cam?.y ?? 0.5, cam?.sx ?? 1, 0]));
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: tex.createView(), loadOp: "clear", storeOp: "store", clearValue: [1, 1, 1, 1] }] });
    pass.setPipeline(pBg); pass.setBindGroup(0, gBg); pass.draw(3);
    if (n > 0) { pass.setPipeline(pSp); pass.setBindGroup(0, gSp); pass.draw(6, n); }
    nd = Math.min(nd || 0, nDet);
    if (nd > 0 && gDt) { pass.setPipeline(pDt); pass.setBindGroup(0, gDt); pass.draw(6, nd); }
    pass.end();
  }
  return {
    format: FT,
    draw(view, w, h, n, cam = null, nd = 0) {
      const enc = dev.createCommandEncoder();
      paint(enc, w, h, n, cam, nd);
      const pass = enc.beginRenderPass({ colorAttachments: [{ view, loadOp: "clear", storeOp: "store", clearValue: [1, 1, 1, 1] }] });
      pass.setPipeline(pCp); pass.setBindGroup(0, gCp); pass.draw(3); pass.end();
      dev.queue.submit([enc.finish()]);
    },
    // the float picture as RGBA floats, w * h * 4 (what the screen shows before its 8-bit rounding)
    async readback(w, h, n, cam = null, nd = 0) {
      const enc = dev.createCommandEncoder();
      paint(enc, w, h, n, cam, nd);
      const bpp = FT === "rgba32float" ? 16 : 8, bpr = Math.ceil(w * bpp / 256) * 256;
      const buf = dev.createBuffer({ size: bpr * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr }, [w, h]);
      dev.queue.submit([enc.finish()]);
      await buf.mapAsync(GPUMapMode.READ);
      const raw = buf.getMappedRange(), out = new Float32Array(w * h * 4);
      const half = (v) => { const e = (v >> 10) & 31, m = v & 1023, s = v >> 15 ? -1 : 1;
        return e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15); };
      for (let y = 0; y < h; y++) {
        if (bpp === 16) out.set(new Float32Array(raw, y * bpr, w * 4), y * w * 4);
        else { const r = new Uint16Array(raw, y * bpr, w * 4); for (let i = 0; i < w * 4; i++) out[y * w * 4 + i] = half(r[i]); }
      }
      buf.unmap(); buf.destroy();
      return out;
    },
    // detail splats (12 floats each) and tile clip rectangles (4 floats each); replaces what was there
    setDetail(spl, rects) {
      for (const b of dBufs) b.destroy();
      dBufs = []; gDt = null; nDet = spl.length / 12;
      if (!nDet) return;
      if (!pDt) { const m = dev.createShaderModule({ code: DETAIL }); pDt = mk(m, "vd", "fd", FT, OVER); }
      const sb = (a) => { const b = dev.createBuffer({ size: Math.max(16, a.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); dev.queue.writeBuffer(b, 0, a); dBufs.push(b); return b; };
      const bs = sb(spl), br = sb(rects);
      gDt = dev.createBindGroup({ layout: pDt.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: ub } }, { binding: 2, resource: { buffer: bs } }, { binding: 3, resource: { buffer: br } }] });
    },
    destroy() { tex?.destroy(); ub.destroy(); for (const b of dBufs) b.destroy(); dBufs = []; tex = null; tw = th = 0; },
  };
}
