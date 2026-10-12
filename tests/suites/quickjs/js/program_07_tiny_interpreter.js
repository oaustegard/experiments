function tokenize(src) { return src.match(/\d+\.?\d*|[-+*/()^]/g); }
function parse(tokens) { let p = 0; const peek = () => tokens[p], next = () => tokens[p++];
  function prim() { const t = next(); if (t === '(') { const v = expr(); next(); return v; } if (t === '-') return -prim(); return parseFloat(t); }
  function pow() { const b = prim(); if (peek() === '^') { next(); return b ** pow(); } return b; }
  function term() { let v = pow(); while (peek() === '*' || peek() === '/') v = next() === '*' ? v * pow() : v / pow(); return v; }
  function expr() { let v = term(); while (peek() === '+' || peek() === '-') v = next() === '+' ? v + term() : v - term(); return v; } return expr(); }
for (const e of ['1+2*3', '(1+2)*3', '2^3^2', '-4+10/4', '3.5*2-1', '((2))*((3+4))']) console.log(e, '=', parse(tokenize(e)));
