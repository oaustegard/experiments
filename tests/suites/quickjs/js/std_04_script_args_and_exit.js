import * as std from 'std';
console.log(scriptArgs.length, typeof scriptArgs[0]);
globalThis.exitSoon = () => std.exit(7); console.log('before'); setTimeout(() => { console.log('exiting'); std.exit(7); }, 1);
