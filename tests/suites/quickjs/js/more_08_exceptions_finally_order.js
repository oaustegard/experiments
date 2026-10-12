function f() { const log = []; try { try { throw 1; } finally { log.push('f1'); } } catch (e) { log.push('c' + e); } finally { log.push('f2'); } return log; } console.log(f());
function g() { try { throw new Error('a'); } catch (e) { try { throw new Error('b'); } catch (e2) { return e.message + e2.message; } finally { console.log('inner finally'); } } } console.log(g());
function h() { l: try { return 'ret'; } finally { break l; } return 'after'; } console.log(h());
function k() { for (const x of [1, 2, 3]) { try { if (x === 2) continue; if (x === 3) return x; } finally { console.log('fin', x); } } } console.log(k());
const r = (() => { try { return 'a'; } finally { console.log('side'); } })(); console.log(r);
