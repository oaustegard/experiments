import * as std from 'std';
import * as os from 'os';
std.printf('%d %s %5.2f %x|\n', 42, 'str', 3.14159, 255);
console.log(std.sprintf('%05d|%-5s|%c', 42, 'ab', 65), typeof std.getenv('PATH'), std.getenv('HCC_NOPE_UNSET'), typeof os.getcwd()[0], typeof os.platform);
console.log(std.parseExtJSON('{a:1, b:[1,2,], "c":0x10, d:.5}'), std.strerror(2), std.loadFile('/nonexistent'), typeof std.evalScript);
std.puts('puts\n'); std.out.puts('out\n'); std.out.printf('%s\n', 'printf'); std.err.puts('');
