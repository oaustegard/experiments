let keep = []; for (let i = 0; i < 20000; i++) { const o = { i, a: [i, i + 1], s: 'v' + i }; o.self = o; if (i % 100 === 0) keep.push(o); }
console.log(keep.length, keep[199].s, keep[5].self === keep[5]); const big = []; for (let i = 0; i < 300; i++) big.push(new Array(1000).fill(i)); console.log(big.length, big[299][999]);
let str = ''; for (let i = 0; i < 2000; i++) str += i; console.log(str.length); const m = new Map(); for (let i = 0; i < 5000; i++) m.set('k' + i, { i }); for (let i = 0; i < 5000; i += 2) m.delete('k' + i); console.log(m.size);
