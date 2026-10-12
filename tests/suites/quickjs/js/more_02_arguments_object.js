function f(a, b) { arguments[0] = 'changed'; b = 'B'; return [a, arguments[1], arguments.length, typeof arguments.callee]; } console.log(f(1, 2), f(1));
function g(a, b = 2) { arguments[0] = 'x'; return [a, arguments.length]; } console.log(g(1));
function h() { 'use strict'; return [typeof arguments.callee === 'undefined' ? 'noncallee' : 'callee']; } try { console.log(h(1)); } catch (e) { console.log(e.name); }
console.log((function () { return [].slice.call(arguments, 1); })(1, 2, 3), (function () { return Array.from(arguments).length; })(...'abcd'));
