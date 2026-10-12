import * as os from 'os';
import * as std from 'std';
let n = 0; const t = os.setTimeout(() => console.log('t1'), 1); os.setTimeout(() => { console.log('t2'); n++; }, 3); os.clearTimeout(t);
console.log(typeof os.now(), os.getcwd() === os.realpath('.')[0], os.isatty(0), typeof os.pid);
const r = std.popen('echo hello; echo err >&2', 'r'); console.log(r.readAsString().trim()); console.log(r.close());
console.log(os.exec(['sh', '-c', 'exit 3']), os.exec(['true']));
const fd = os.open('/dev/null', os.O_RDWR); console.log(fd >= 0, os.write(fd, new Uint8Array(4).buffer, 0, 4), os.close(fd));
