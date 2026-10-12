import * as std from 'std';
export const x = 1; export default function f() { return 2; } export { x as y };
console.log(x, typeof f);
console.log(typeof import.meta, typeof import.meta.url, import.meta.main);
