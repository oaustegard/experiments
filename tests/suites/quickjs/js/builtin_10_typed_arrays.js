const buf = new ArrayBuffer(16); const u8 = new Uint8Array(buf), i32 = new Int32Array(buf, 4, 2), dv = new DataView(buf);
i32[0] = -2; i32[1] = 0x01020304; console.log(u8.slice(4, 12), dv.getInt32(4, true), dv.getUint32(8, false), dv.getFloat32(4, true), buf.byteLength, i32.byteOffset, i32.length);
dv.setFloat64(8, Math.PI); console.log(dv.getFloat64(8), [...u8.slice(8)], dv.getBigInt64(8) > 0n, dv.getUint16(8), dv.getInt8(8));
console.log(new Uint8Array([255, 256, -1, 1.9]), new Int8Array([127, 128, 255]), new Uint8ClampedArray([300, -5, 1.5, 2.5]), new Float32Array([1.1])[0], new Float64Array(2), Uint16Array.of(1, 2), Uint8Array.from('123'));
const t = new Int16Array([5, 1, 4]); t.sort(); console.log(t, t.map(x => x * 2), t.reduce((a, b) => a + b), Array.from(t), t.subarray(1).length, t.fill(7, 1), t.reverse(), t.indexOf(7), t.join('|'), t.BYTES_PER_ELEMENT, Object.prototype.toString.call(t));
console.log(new BigInt64Array(2).fill(-1n), new BigUint64Array([2n ** 64n - 1n])[0], new Uint32Array(new Float32Array([1.5, 2.5, -1])), new Float32Array([16777217])[0]);
const r = new ArrayBuffer(8, { maxByteLength: 16 }); console.log(r.resizable, r.maxByteLength); r.resize?.(12); console.log(r.byteLength, buf.slice(4, 8).byteLength, ArrayBuffer.isView(u8), new Uint8Array(buf.slice(0, 2)).length);
try { new Int32Array(buf, 1); } catch (e) { console.log(e.name); }
