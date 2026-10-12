console.log('start');
setTimeout(() => { throw new RangeError('uncaught in timer'); }, 1);
