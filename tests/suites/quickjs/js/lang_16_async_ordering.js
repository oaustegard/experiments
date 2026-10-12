console.log('1 sync');
setTimeout(() => console.log('8 timeout'), 0);
Promise.resolve().then(() => console.log('4 micro1')).then(() => console.log('6 micro2'));
queueMicrotask?.(() => console.log('5 qm'));
(async () => { console.log('2 async start'); await null; console.log('5b after await'); await new Promise(r => setTimeout(r, 1)); console.log('9 after timer'); })();
new Promise(r => { console.log('3 executor'); r(); }).then(() => console.log('5c then'));
console.log('3b sync end');
