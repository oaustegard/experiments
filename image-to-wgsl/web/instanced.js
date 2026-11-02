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
// Bindings: @group(0) @binding(0) uniform {a: vec4f(width, height, zoom, 0), b: vec4f(x, y, 0, 0)}, where (x, y)
// is the image point (as a fraction of its width and height) at the centre of the view; @binding(1) the
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
    `fn kk()->f32{return min(u.a.x/A,u.a.y)*u.a.z;}fn oo()->vec2f{return u.a.xy*.5-vec2f(u.b.x*A,u.b.y)*kk();}` +
    `fn inside(p:vec2f)->bool{let q=(p-oo())/kk();return!(q.x<0.||q.y<0.||q.x>A||q.y>1.);}` +
    `fn tri(i:u32)->vec4f{return vec4f(vec2f(f32(i%2u),f32(i/2u))*4.-1.,0.,1.);}` +
    `@vertex fn vb(@builtin(vertex_index)i:u32)->@builtin(position)vec4f{return tri(i);}` +
    `@fragment fn fb(@builtin(position)p:vec4f)->@location(0)vec4f{if(!inside(p.xy)){return vec4f(1.);}let h=((p.x-oo().x)/kk())/A;let y=(p.y-oo().y)/kk();` +
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

// Host side. draw(view, w, h, n, cam) paints the first n splats into `view` (w x h pixels, format `fmt`);
// cam = {z, x, y} zooms by z about image point (x, y) (fractions of its width and height), default fit.
// Pass a device created with the "float32-blendable" feature, when the adapter has it, for exact blending.
// Kept free of imports so a downloaded page can embed it with createRenderer.toString().
export function createRenderer(dev, fmt, code) {
  const mod = dev.createShaderModule({ code });
  const ub = dev.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const OVER = { color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
                 alpha: { srcFactor: "zero", dstFactor: "one", operation: "add" } };
  const mk = (v, f, format, blend) => dev.createRenderPipeline({ layout: "auto", vertex: { module: mod, entryPoint: v },
    fragment: { module: mod, entryPoint: f, targets: [{ format, blend }] }, primitive: { topology: "triangle-list" } });
  const FT = dev.features.has("float32-blendable") ? "rgba32float" : "rgba16float";
  const pBg = mk("vb", "fb", FT), pSp = mk("vs", "fs", FT, OVER), pCp = mk("vc", "fc", fmt);
  const ubg = (p) => dev.createBindGroup({ layout: p.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: ub } }] });
  const gBg = ubg(pBg), gSp = ubg(pSp);
  let tex = null, gCp = null, tw = 0, th = 0;
  return {
    format: FT,
    draw(view, w, h, n, cam = null) {
      if (w !== tw || h !== th) {
        tex?.destroy();
        tex = dev.createTexture({ size: [w, h], format: FT, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
        gCp = dev.createBindGroup({ layout: pCp.getBindGroupLayout(0), entries: [{ binding: 1, resource: tex.createView() }] });
        tw = w; th = h;
      }
      dev.queue.writeBuffer(ub, 0, new Float32Array([w, h, cam?.z ?? 1, 0, cam?.x ?? 0.5, cam?.y ?? 0.5, 0, 0]));
      const enc = dev.createCommandEncoder();
      let pass = enc.beginRenderPass({ colorAttachments: [{ view: tex.createView(), loadOp: "clear", storeOp: "store", clearValue: [1, 1, 1, 1] }] });
      pass.setPipeline(pBg); pass.setBindGroup(0, gBg); pass.draw(3);
      if (n > 0) { pass.setPipeline(pSp); pass.setBindGroup(0, gSp); pass.draw(6, n); }
      pass.end();
      pass = enc.beginRenderPass({ colorAttachments: [{ view, loadOp: "clear", storeOp: "store", clearValue: [1, 1, 1, 1] }] });
      pass.setPipeline(pCp); pass.setBindGroup(0, gCp); pass.draw(3); pass.end();
      dev.queue.submit([enc.finish()]);
    },
    destroy() { tex?.destroy(); ub.destroy(); tex = null; tw = th = 0; },
  };
}
