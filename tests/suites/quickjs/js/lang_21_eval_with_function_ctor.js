console.log(eval('1 + 2'), eval('var ev = 5; ev * 2'), typeof ev, (0, eval)('typeof ev'));
const g = new Function('return this')(); console.log(typeof g, g === globalThis);
globalThis.gv = 7; console.log(gv, typeof window, typeof globalThis.console, typeof require, typeof std);
console.log(eval('({a:1})').a, eval('[1,2]').length, eval('(function(){ return arguments.length })(1,2,3)'));
try { eval('let let = 1'); } catch (e) { console.log(e.name); }
console.log(Function.prototype.toString.call(function a(b) { return b; }), (class Z {}).toString());
