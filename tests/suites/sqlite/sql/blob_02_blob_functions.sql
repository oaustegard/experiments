SELECT hex('abc'), unhex('616263'), unhex('zz'), unhex('6'), unhex('61 62', ' '), hex(unhex('00ff')), hex(char(0, 255)), typeof(unhex('61'));
SELECT instr(x'00112233', x'1122'), length(x'00'), hex(randomblob(3)) GLOB '[0-9A-F][0-9A-F][0-9A-F][0-9A-F][0-9A-F][0-9A-F]';
SELECT hex(CAST('é' AS BLOB)), CAST(x'c3a9' AS TEXT), length(CAST(x'c3a9' AS TEXT)), length(x'c3a9');
