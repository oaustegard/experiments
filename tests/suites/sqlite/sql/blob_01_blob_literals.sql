SELECT x'48656C6C6F', hex(x'00ff'), length(x'0102030405'), typeof(x''), x'' = '', X'ab' = x'AB';
SELECT hex(zeroblob(4)), quote(x'deadbeef'), substr(x'0102030405', 2, 3), hex(substr(x'0102030405', 2, 3)), x'01' || x'02', hex(x'01' || x'02');
