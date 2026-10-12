var a = 1; { var a = 2; } console.log(a);
let b = 1; { let b = 2; console.log(b); } console.log(b);
console.log(typeof hoisted, hoisted2()); var hoisted = 1; function hoisted2() { return 'f'; }
try { console.log(tdz); let tdz = 1; } catch (e) { console.log(e.name, e.message); }
const c = [1]; c.push(2); console.log(c); try { c = 1; } catch (e) { console.log(e.name); }
for (var i = 0; i < 3; i++) setTimeout(() => {}, 0); console.log(i);
var fs = []; for (let j = 0; j < 3; j++) fs.push(() => j); console.log(fs.map(f => f()));
