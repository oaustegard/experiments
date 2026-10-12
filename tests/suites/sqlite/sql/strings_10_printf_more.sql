SELECT printf('%lld', 9223372036854775807), printf('%.0f', 0.5), printf('%.0f', 1.5), printf('%.3s', 'abcdef'), printf('%*d', 6, 42), printf('%-*d|', 6, 42);
SELECT printf('%,d', 1234567), printf('%!.3f', 1.0), printf('%d %d', 1), printf('%z', 1), printf('%i', 5), printf('%u', -1);
SELECT printf('%.15g', 0.1), printf('%.17g', 0.1), printf('%f', 1e20), printf('%e', 0), printf('%.2f', -0.001);
