console.log('a');
Promise.reject(new TypeError('unhandled rejection'));
console.log('b');
