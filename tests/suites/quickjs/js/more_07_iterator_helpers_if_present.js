const it = [1, 2, 3, 4].values(); console.log(typeof it.map, typeof Iterator);
if (typeof Iterator !== 'undefined' && it.map) { console.log(typeof it.take, typeof Iterator.from); }
console.log(Object.prototype.toString.call([].values()), Object.prototype.toString.call(new Map().entries()), Object.prototype.toString.call(''[Symbol.iterator]()), typeof [][Symbol.iterator]().next);
