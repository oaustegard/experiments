// Walk a snappy block-format stream and count bytes emitted as literals vs
// bytes emitted by back-reference copies. This is what decides whether the
// wide-copy change can matter on a given page.
export function snappyMix(buf) {
  let i = 0, shift = 0, uncompressed = 0
  for (;;) { const b = buf[i++]; uncompressed |= (b & 0x7f) << shift; shift += 7; if (!(b & 0x80)) break }
  let literal = 0, copy = 0, nLit = 0, nCopy = 0
  const copyLens = []
  while (i < buf.length) {
    const tag = buf[i]
    const type = tag & 0x03
    if (type === 0) { // literal
      let len = tag >> 2
      i += 1
      if (len >= 60) { const extra = len - 59; len = 0
        for (let k = 0; k < extra; k++) len |= buf[i + k] << (8 * k)
        i += extra }
      len += 1
      literal += len; nLit++; i += len
    } else {
      let len, off
      if (type === 1) { len = ((tag >> 2) & 0x07) + 4; off = ((tag >> 5) << 8) | buf[i + 1]; i += 2 }
      else if (type === 2) { len = (tag >> 2) + 1; off = buf[i+1] | (buf[i+2] << 8); i += 3 }
      else { len = (tag >> 2) + 1; off = buf[i+1] | (buf[i+2]<<8) | (buf[i+3]<<16) | (buf[i+4]<<24); i += 5 }
      copy += len; nCopy++; copyLens.push(len)
    }
  }
  copyLens.sort((a,b)=>a-b)
  return { uncompressed, literal, copy, nLit, nCopy,
    literal_pct: +(100*literal/(literal+copy)).toFixed(1),
    mean_literal_run: +(literal/Math.max(nLit,1)).toFixed(1),
    mean_copy_len: +(copy/Math.max(nCopy,1)).toFixed(1),
    median_copy_len: copyLens[Math.floor(copyLens.length/2)] ?? 0 }
}
