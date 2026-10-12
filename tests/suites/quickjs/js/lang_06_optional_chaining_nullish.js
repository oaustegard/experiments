const o = { a: { b: [1, { c: 2 }] }, f() { return this.a; }, n: null, z: 0, e: '' };
console.log(o?.a?.b?.[1]?.c, o.x?.y.z, o.n?.x, o.f?.().b.length, o.g?.(), o?.['a']?.b?.length);
console.log(o.n ?? 'd', o.z ?? 'd', o.z || 'd', o.e ?? 'd', undefined ?? null ?? 3);
let a = null; a ??= 5; let b = 0; b ||= 7; let c = 1; c &&= 9; console.log(a, b, c);
delete o?.a.b; console.log(o.a);
