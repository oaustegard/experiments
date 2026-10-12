SELECT sqlite_compileoption_used('SOUNDEX'), 1;
SELECT lower('ÀÉ'), upper('àé'), length(char(0)||'a'), unicode(''), zeroblob(3), hex(zeroblob(2));
SELECT sqlite_version() GLOB '3.*', typeof(sqlite_source_id()), sqlite_compileoption_used('ENABLE_FTS5'), sqlite_compileoption_used('NOPE');
