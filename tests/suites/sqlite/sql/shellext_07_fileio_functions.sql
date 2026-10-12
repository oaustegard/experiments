SELECT writefile('qjs_fileio.txt', 'line1' || char(10) || 'line2');
SELECT readfile('qjs_fileio.txt'), length(readfile('qjs_fileio.txt'));
SELECT writefile('qjs_fileio.bin', x'00010203');
SELECT hex(readfile('qjs_fileio.bin')), readfile('nonexistent.file') IS NULL;
SELECT count(*) FROM fsdir('.') WHERE name LIKE 'qjs_fileio%';
SELECT lsmode(33188), lsmode(16877);
SELECT name, mode FROM fsdir('.') WHERE name = 'qjs_fileio.bin';
.shell rm -f qjs_fileio.txt qjs_fileio.bin
