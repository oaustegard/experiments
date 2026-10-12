function f() { try { throw new TypeError('bad'); } catch (e) { return e instanceof TypeError && e.message; } finally { console.log('fin'); } }
console.log(f());
function g() { try { return 'try'; } finally { console.log('cleanup'); } } console.log(g());
function h() { for (;;) { try { break; } finally { console.log('brk'); } } return 1; } console.log(h());
try { null.x; } catch (e) { console.log(e.constructor.name, e instanceof Error); }
try { undefinedFn(); } catch (e) { console.log(e.name + ': ' + e.message); }
try { throw { code: 42 }; } catch ({ code }) { console.log(code); }
class MyErr extends Error { constructor(m) { super(m); this.name = 'MyErr'; } } try { throw new MyErr('custom'); } catch (e) { console.log(String(e), e instanceof MyErr, e instanceof Error); }
const e = new Error('outer', { cause: new Error('inner') }); console.log(e.cause.message);
try { JSON.parse('{bad'); } catch (e) { console.log(e.name); }
try { new Array(-1); } catch (e) { console.log(e.name, e.message); }
try { (function r() { r(); })(); } catch (e) { console.log(e.name); }
