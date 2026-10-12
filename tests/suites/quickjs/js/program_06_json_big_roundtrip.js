const data = Array.from({ length: 500 }, (_, i) => ({ id: i, name: 'item' + i, tags: ['t' + (i % 7), 't' + (i % 3)], v: i / 7, nested: { a: [i, i * 2], ok: i % 2 === 0 } }));
const s = JSON.stringify(data); const back = JSON.parse(s); console.log(s.length, back.length, back[499].nested, JSON.stringify(back) === s, back.filter(x => x.nested.ok).length);
