var o = { a: 1, b: 2 }; with (o) { a = 10; var c = a + b; } console.log(o, c, typeof a);
function f() { with ({ x: 5 }) { return function () { return x; }; } } console.log(f()());
