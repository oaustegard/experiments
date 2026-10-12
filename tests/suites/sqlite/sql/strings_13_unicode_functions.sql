SELECT unicode('😀'), char(128512), length('😀'), octet_length('😀'), hex('😀'), upper('ß'), lower('İ'), length(char(65,0,66));
SELECT 'ä' < 'z', 'ä' COLLATE NOCASE = 'Ä', 'Ä' COLLATE BINARY = 'ä', 'a ' COLLATE RTRIM = 'a';
