import * as std from 'std';
import * as os from 'os';
const name = 'qjs_test_file.txt'; const f = std.open(name, 'w'); f.puts('line1\nline2\n'); f.printf('%d\n', 3); f.close();
const g = std.open(name, 'r'); console.log(g.getline(), g.readAsString().length, g.eof(), g.tell()); g.seek(0, std.SEEK_SET); console.log(g.readAsString(5), g.getByte()); g.close();
console.log(std.loadFile(name).split('\n'), os.stat(name)[0].size, os.readdir('.')[0].includes(name)); os.remove(name);
console.log(os.stat(name)[1] !== 0, std.open(name, 'r'), std.loadFile(name));
const [err] = [os.mkdir('qd')]; console.log(err, os.readdir('qd')[0], os.rename('qd', 'qd2'), os.remove('qd2'), os.stat('.')[0].mode !== 0);
