# SQL test scripts, part B (JSON, FTS5, functions, dates, math, affinity, transactions,
# pragmas, blobs, collations, rtree, errors).  Same format as sqlite_scripts_a.py.

SCRIPTS = {}

SCRIPTS["json"] = [
("json_valid_type", """
SELECT json_valid('{"a":1}'), json_valid('{a:1}'), json_valid(''), json_valid('[1,2,'), json_valid('null');
SELECT json_type('{"a":[1,2.5,"x",null,true]}'), json_type('{"a":[1,2.5,"x",null,true]}', '$.a'), json_type('[1,2]', '$[1]');
SELECT json_type('[1,2.5,"x",null,true,false]','$[0]'), json_type('[1,2.5,"x",null,true,false]','$[1]'), json_type('[1,2.5,"x",null,true,false]','$[2]'), json_type('[1,2.5,"x",null,true,false]','$[3]'), json_type('[1,2.5,"x",null,true,false]','$[4]'), json_type('[1,2.5,"x",null,true,false]','$[5]');
"""),
("json_extract", """
SELECT json_extract('{"a":1,"b":{"c":[10,20,30],"d":"s"}}', '$.a');
SELECT json_extract('{"a":1,"b":{"c":[10,20,30],"d":"s"}}', '$.b.c[1]', '$.b.d');
SELECT json_extract('{"a":1,"b":{"c":[10,20,30],"d":"s"}}', '$.b.c');
SELECT json_extract('{"a":1}', '$.zz'), json_extract('[1,2,3]', '$[#-1]');
SELECT '{"a":{"b":5}}' -> '$.a', '{"a":{"b":5}}' ->> '$.a.b', '[10,20]' ->> 1, '{"k":"v"}' -> 'k';
"""),
("json_object_array", """
SELECT json_object('a',1,'b','two','c',NULL,'d',json_array(1,2)), json_array(1,'x',NULL,2.5,json('{"q":1}'));
SELECT json_array(), json_object(), json_quote('a"b'), json_quote(3), json_quote(NULL);
SELECT json_array_length('[1,[2,3],4]'), json_array_length('[1,[2,3],4]','$[1]'), json_array_length('{"a":1}');
"""),
("json_modify", """
SELECT json_set('{"a":1}', '$.b', 2, '$.a', 10);
SELECT json_insert('{"a":1}', '$.a', 99, '$.c', 3);
SELECT json_replace('{"a":1}', '$.a', 'new', '$.zz', 1);
SELECT json_remove('{"a":1,"b":2,"c":[1,2,3]}', '$.b', '$.c[0]');
SELECT json_patch('{"a":1,"b":{"c":2}}', '{"b":{"c":null,"d":4},"e":5}');
SELECT json_set('[1,2,3]', '$[#]', 4), json_set('{}', '$.a.b', 1);
"""),
("json_each_tree", """
SELECT key, value, type, atom, fullkey, path FROM json_each('{"a":1,"b":[true,null],"c":{"d":"e"}}');
SELECT key, value, type FROM json_tree('{"a":[1,{"b":2}]}') ;
SELECT fullkey FROM json_tree('{"a":[1,{"b":2}]}') WHERE type NOT IN ('object','array');
"""),
("json_each_table", """
CREATE TABLE doc(id INTEGER PRIMARY KEY, j TEXT);
INSERT INTO doc VALUES(1,'{"tags":["a","b"],"n":1}'),(2,'{"tags":["b","c","d"],"n":2}'),(3,'{"n":3}');
SELECT doc.id, e.value FROM doc, json_each(doc.j, '$.tags') e ORDER BY 1,2;
SELECT e.value, count(*) FROM doc, json_each(doc.j, '$.tags') e GROUP BY 1 ORDER BY 2 DESC, 1;
SELECT id, json_extract(j,'$.n') FROM doc WHERE json_extract(j,'$.tags[0]') = 'b';
"""),
("json_group", """
CREATE TABLE t(g, k, v);
INSERT INTO t VALUES('x','a',1),('x','b',2),('y','c',3);
SELECT g, json_group_array(v), json_group_object(k, v) FROM t GROUP BY g ORDER BY g;
SELECT json_group_array(json_object('k',k,'v',v)) FROM t;
"""),
("json_minify_pretty", """
SELECT json('  { "a" : [ 1 , 2 ,3 ] , "b" : "x y" }  ');
SELECT json_pretty('{"a":[1,2,{"b":null}],"c":"d"}');
SELECT json('[1,2'), 1;
"""),
("json_nested_types", """
SELECT json_extract('{"a":1.0}', '$.a'), json_extract('{"a":1e3}', '$.a'), json_extract('{"a":12345678901234567890}', '$.a');
SELECT json_extract('{"a":"\\u00e9\\n"}', '$.a'), json_quote('é'), json('{"a":"\\u00e9"}');
SELECT typeof(json_extract('{"a":true}','$.a')), json_extract('{"a":true}','$.a'), json_extract('{"a":null}','$.a') IS NULL;
"""),
("json_array_ops", """
SELECT json_array_length(json_group_array(value)) FROM json_each('[5,6,7]');
SELECT json_insert('[1,2]', '$[1]', 'x'), json_set('[1,2]', '$[5]', 'x'), json_remove('[1,2,3]', '$[1]');
SELECT value FROM json_each('[[1,2],[3,4]]') ;
SELECT sum(value) FROM json_tree('[[1,2],[3,[4,5]]]') WHERE type='integer';
"""),
("json_errors", """
SELECT json_extract('not json', '$.a');
SELECT json_extract('{"a":1}', 'bad path');
SELECT json_array_length('{');
SELECT json_object('a');
SELECT json_set('{"a":1}', '$.a');
"""),
("jsonb_basic", """
SELECT json(jsonb('{"a":[1,2,3],"b":"x"}'));
SELECT jsonb_extract('{"a":[1,2,3]}', '$.a') IS NOT NULL, json_extract(jsonb('{"a":[1,2,3]}'), '$.a[2]');
SELECT json(jsonb_set(jsonb('{"a":1}'), '$.b', 2)), json(jsonb_array(1,2,'x')), json(jsonb_object('k', 1));
SELECT hex(jsonb('{"a":1}')) <> '', typeof(jsonb('[1]'));
"""),
("json_in_where_index", """
CREATE TABLE ev(id INTEGER PRIMARY KEY, body TEXT);
INSERT INTO ev(body) SELECT json_object('type', CASE WHEN value%3=0 THEN 'a' ELSE 'b' END, 'n', value) FROM json_each('[1,2,3,4,5,6,7,8,9,10]');
CREATE INDEX evt ON ev(json_extract(body,'$.type'));
SELECT count(*) FROM ev WHERE json_extract(body,'$.type')='a';
SELECT sum(json_extract(body,'$.n')) FROM ev WHERE body->>'type'='b';
"""),
]

SCRIPTS["fts5"] = [
("fts5_basic", """
CREATE VIRTUAL TABLE docs USING fts5(title, body);
INSERT INTO docs VALUES('cats','the quick brown fox jumps over the lazy dog'),
 ('dogs','a lazy dog sleeps all day'),('foxes','the red fox and the grey fox'),('misc','nothing to see here');
SELECT title FROM docs WHERE docs MATCH 'fox' ORDER BY rank;
SELECT title FROM docs WHERE docs MATCH 'lazy AND dog' ORDER BY title;
SELECT title FROM docs WHERE body MATCH 'quick OR sleeps' ORDER BY title;
"""),
("fts5_phrase_near", """
CREATE VIRTUAL TABLE d USING fts5(t);
INSERT INTO d VALUES('one two three four five'),('five four three two one'),('one three five'),('two one');
SELECT rowid FROM d WHERE d MATCH '"one two"' ORDER BY rowid;
SELECT rowid FROM d WHERE d MATCH 'NEAR(one five, 3)' ORDER BY rowid;
SELECT rowid FROM d WHERE d MATCH 'one NOT two' ORDER BY rowid;
SELECT rowid FROM d WHERE d MATCH 'th*' ORDER BY rowid;
"""),
("fts5_highlight_snippet", """
CREATE VIRTUAL TABLE d USING fts5(t);
INSERT INTO d VALUES('The quick brown fox jumps over the lazy dog'),('Pack my box with five dozen liquor jugs');
SELECT highlight(d, 0, '[', ']') FROM d WHERE d MATCH 'fox OR box';
SELECT snippet(d, 0, '<', '>', '...', 4) FROM d WHERE d MATCH 'lazy';
"""),
("fts5_bm25_rank", """
CREATE VIRTUAL TABLE d USING fts5(t);
INSERT INTO d VALUES('apple'),('apple apple banana'),('banana cherry'),('apple banana cherry apple apple');
SELECT rowid, round(bm25(d),4) FROM d WHERE d MATCH 'apple' ORDER BY bm25(d), rowid;
SELECT rowid FROM d WHERE d MATCH 'apple' ORDER BY rank, rowid;
"""),
("fts5_content_external", """
CREATE TABLE src(id INTEGER PRIMARY KEY, txt);
CREATE VIRTUAL TABLE idx USING fts5(txt, content='src', content_rowid='id');
INSERT INTO src VALUES(1,'alpha beta'),(2,'beta gamma'),(3,'gamma delta');
INSERT INTO idx(rowid, txt) SELECT id, txt FROM src;
SELECT rowid FROM idx WHERE idx MATCH 'beta' ORDER BY rowid;
INSERT INTO idx(idx) VALUES('integrity-check');
SELECT count(*) FROM idx WHERE idx MATCH 'delta OR alpha';
"""),
("fts5_tokenizers", """
CREATE VIRTUAL TABLE a USING fts5(t, tokenize='porter unicode61');
CREATE VIRTUAL TABLE b USING fts5(t, tokenize='unicode61 remove_diacritics 2');
CREATE VIRTUAL TABLE c USING fts5(t, tokenize='trigram');
INSERT INTO a VALUES('running runs ran'),('jumped jumping');
INSERT INTO b VALUES('Café résumé'),('naive');
INSERT INTO c VALUES('hello world'),('help wanted');
SELECT rowid FROM a WHERE a MATCH 'run' ORDER BY 1;
SELECT rowid FROM a WHERE a MATCH 'jump' ORDER BY 1;
SELECT rowid FROM b WHERE b MATCH 'cafe' ORDER BY 1;
SELECT rowid FROM c WHERE c MATCH 'ell' ORDER BY 1;
"""),
("fts5_delete_update", """
CREATE VIRTUAL TABLE d USING fts5(a, b);
INSERT INTO d VALUES('x y','z'),('y z','x'),('q','r');
UPDATE d SET a='changed' WHERE rowid=1;
DELETE FROM d WHERE rowid=3;
SELECT rowid, * FROM d ORDER BY rowid;
SELECT rowid FROM d WHERE d MATCH 'a:changed OR b:x' ORDER BY rowid;
SELECT rowid FROM d WHERE d MATCH '{a b}:y' ORDER BY rowid;
"""),
("fts5_vocab", """
CREATE VIRTUAL TABLE d USING fts5(t);
INSERT INTO d VALUES('a b c'),('a b'),('a');
CREATE VIRTUAL TABLE v USING fts5vocab(d, 'row');
SELECT term, doc FROM v ORDER BY term;
CREATE VIRTUAL TABLE v2 USING fts5vocab(d, 'col');
SELECT term, col, doc, cnt FROM v2 ORDER BY term;
"""),
]

SCRIPTS["strings"] = [
("length_substr", """
SELECT length('hello'), length(''), length(NULL), length('héllo'), length(x'0102'), length(12345);
SELECT substr('hello',2), substr('hello',2,3), substr('hello',-3), substr('hello',-3,2), substr('hello',0), substr('hello',10), substr('hello',2,-1);
SELECT substring('hello world', 7, 5), octet_length('héllo');
"""),
("case_trim", """
SELECT upper('abc'), lower('ABC'), upper('héllo'), trim('  x  '), ltrim('  x  '), rtrim('  x  '), trim('xxhixx','x'), ltrim('abcabc','ab'), rtrim('hello!!','!');
"""),
("replace_instr", """
SELECT replace('hello world','o','0'), replace('aaa','a','bb'), replace('abc','','x'), replace('abc','b',''), replace(NULL,'a','b');
SELECT instr('hello','l'), instr('hello','z'), instr('hello',''), instr(x'010203',x'0203'), instr('héllo','l');
"""),
("concat_functions", """
SELECT 'a'||'b', 'a'||NULL, concat('a',NULL,1,2.5), concat_ws('-','a',NULL,'b'), concat_ws(',',1,2,3);
SELECT 1||2, 1.5||'x', hex('abc'), unhex('616263'), char(72,105,128512), unicode('A'), unicode('€');
"""),
("quote_typeof", """
SELECT quote('it''s'), quote(1), quote(1.5), quote(NULL), quote(x'ff'), quote('a'||char(10)||'b');
SELECT typeof(1), typeof(1.5), typeof('a'), typeof(x''), typeof(NULL), typeof(1+1.0), typeof('1'+1);
"""),
("like_escape_case", """
SELECT 'ABC' LIKE 'abc', 'ABC' GLOB 'abc', 'abc' GLOB 'a?c', 'abc' GLOB 'a[a-c]c', 'a.c' LIKE 'a.c', 'abc' LIKE 'a_c';
PRAGMA case_sensitive_like=ON;
SELECT 'ABC' LIKE 'abc', 'abc' LIKE 'abc';
SELECT like('a%','abc'), glob('a*','abc'), like('A_','ab','\\');
"""),
("soundex_unicode_misc", """
SELECT sqlite_compileoption_used('SOUNDEX'), 1;
SELECT lower('ÀÉ'), upper('àé'), length(char(0)||'a'), unicode(''), zeroblob(3), hex(zeroblob(2));
SELECT sqlite_version() GLOB '3.*', typeof(sqlite_source_id()), sqlite_compileoption_used('ENABLE_FTS5'), sqlite_compileoption_used('NOPE');
"""),
("string_compare_sort", """
CREATE TABLE t(s); INSERT INTO t VALUES('b'),('B'),('a'),('A'),('10'),('9'),('é'),('_'),(''),(NULL);
SELECT s FROM t ORDER BY s;
SELECT s FROM t ORDER BY s COLLATE NOCASE, s;
SELECT s FROM t ORDER BY s DESC;
SELECT min(s), max(s) FROM t;
"""),
("format_strings", """
SELECT format('%d items', 3), format('%5.1f|%-6s|%+d', 3.14159, 'ab', 5), format('%x %X %o %c', 255, 255, 8, 65);
SELECT format('%q %Q %w', 'it''s', 'it''s', 'a"b'), format('%Q', NULL), format('%s', NULL), format('%%');
SELECT format('%10.3e', 12345.6789), format('%g %g %g', 0.0001, 123456789.0, 1e-10), format('%05d|%-5d|', 42, 42);
"""),
("printf_more", """
SELECT printf('%lld', 9223372036854775807), printf('%.0f', 0.5), printf('%.0f', 1.5), printf('%.3s', 'abcdef'), printf('%*d', 6, 42), printf('%-*d|', 6, 42);
SELECT printf('%,d', 1234567), printf('%!.3f', 1.0), printf('%d %d', 1), printf('%z', 1), printf('%i', 5), printf('%u', -1);
SELECT printf('%.15g', 0.1), printf('%.17g', 0.1), printf('%f', 1e20), printf('%e', 0), printf('%.2f', -0.001);
"""),
("string_building", """
WITH RECURSIVE r(n,s) AS (SELECT 1,'a' UNION ALL SELECT n+1, s||char(97+n) FROM r WHERE n<10) SELECT group_concat(s, ',') FROM r;
SELECT length(replace(hex(randomblob(0)),'0','')), length(zeroblob(1000)), length(printf('%1000s','x')), length(substr(printf('%5000s','y'),4000));
SELECT trim(printf('%20s','pad')), ltrim(printf('%-20s|','pad'),'p');
"""),
("string_split_via_cte", """
WITH RECURSIVE sp(w, rest) AS (SELECT '', 'the,quick,brown,fox,' UNION ALL SELECT substr(rest,1,instr(rest,',')-1), substr(rest,instr(rest,',')+1) FROM sp WHERE rest<>'')
SELECT w FROM sp WHERE w<>'';
SELECT group_concat(upper(substr(w,1,1))||substr(w,2), ' ') FROM (SELECT 'hello' w UNION ALL SELECT 'big' UNION ALL SELECT 'world');
"""),
("unicode_functions", """
SELECT unicode('😀'), char(128512), length('😀'), octet_length('😀'), hex('😀'), upper('ß'), lower('İ'), length(char(65,0,66));
SELECT 'ä' < 'z', 'ä' COLLATE NOCASE = 'Ä', 'Ä' COLLATE BINARY = 'ä', 'a ' COLLATE RTRIM = 'a';
"""),
]

SCRIPTS["datetime"] = [
("date_basic", """
SELECT date('2024-03-15'), time('2024-03-15 13:45:30'), datetime('2024-03-15 13:45:30'), date('2024-03-15 25:00:00');
SELECT date('2024-02-30'), date('2024-13-01'), date(NULL), date('garbage'), date('2024-03-15T10:20:30Z'), datetime('2024-03-15T10:20:30+02:00');
"""),
("date_modifiers", """
SELECT date('2024-03-15','+1 day'), date('2024-03-15','-1 month'), date('2024-01-31','+1 month'), date('2024-02-29','+1 year'), date('2024-03-15','start of month'), date('2024-03-15','start of year');
SELECT date('2024-03-15','weekday 0'), date('2024-03-15','weekday 5'), date('2024-03-15','+10 days','start of month','+1 month','-1 day');
SELECT datetime('2024-03-15 12:00:00','+36 hours'), datetime('2024-03-15 12:00:00','-90 minutes'), time('12:00:00','+3600 seconds'), datetime('2024-03-15','start of day','+1 day','-1 second');
"""),
("julianday_unix", """
SELECT julianday('2000-01-01'), julianday('2000-01-01 12:00:00'), julianday('2024-03-15') - julianday('2024-01-01');
SELECT unixepoch('1970-01-02'), unixepoch('2024-03-15 10:00:00'), datetime(0,'unixepoch'), datetime(1710497000,'unixepoch'), datetime(1710497000.5,'unixepoch','subsec');
SELECT date(2460385.5), datetime(2460385.5), date(2460385.5,'julianday'), datetime(1710497000,'auto'), datetime(2460385.5, 'auto');
"""),
("strftime_formats", """
SELECT strftime('%Y-%m-%d %H:%M:%S', '2024-03-15 13:45:30'), strftime('%d/%m/%y', '2024-03-05'), strftime('%j', '2024-03-15'), strftime('%w %W %U', '2024-03-15');
SELECT strftime('%s', '2024-03-15 00:00:00'), strftime('%f', '2024-03-15 13:45:30.123'), strftime('%J', '2000-01-01 12:00'), strftime('%%|%H|%I|%M|%p', '2024-03-15 15:07:00');
SELECT strftime('%G-W%V-%u', '2024-03-15'), strftime('%e|%k|%l|%P', '2024-03-05 09:05:00'), strftime('%a %A %b %B', '2024-03-15'), strftime('%F %T %R', '2024-03-15 13:45:30');
"""),
("date_arith_table", """
CREATE TABLE ev(id, d);
INSERT INTO ev VALUES(1,'2024-01-15'),(2,'2024-02-20'),(3,'2024-02-29'),(4,'2023-12-31');
SELECT id, d, strftime('%Y-%m', d) ym, cast(strftime('%j', d) AS INT) doy, julianday(d)-julianday('2024-01-01') AS off FROM ev ORDER BY d;
SELECT strftime('%Y-%m', d), count(*) FROM ev GROUP BY 1 ORDER BY 1;
SELECT id FROM ev WHERE d BETWEEN date('2024-02-01') AND date('2024-02-01','+1 month','-1 day') ORDER BY id;
"""),
("date_leap_edges", """
SELECT date('2000-02-29','+1 year'), date('1900-02-28','+1 day'), date('2100-02-28','+1 day'), date('2024-12-31','+1 day'), date('2024-01-01','-1 day');
SELECT date('0000-01-01'), date('9999-12-31'), date('9999-12-31','+1 day'), date('-0001-01-01'), date('0001-01-01','-1 day');
SELECT julianday('9999-12-31'), unixepoch('0000-01-01'), unixepoch('9999-12-31 23:59:59');
"""),
("date_time_only", """
SELECT time('23:59:59','+1 second'), time('00:00:00','-1 second'), datetime('12:30'), date('12:30'), time('12:30:15.678'), time('12:30:15.678','subsec');
SELECT datetime('2024-03-15 12:30:00','localtime','utc') IS NOT NULL, datetime('2024-03-15 12:30:00','utc') IS NOT NULL;
"""),
("date_diff_expressions", """
SELECT cast((julianday('2024-12-25') - julianday('2024-03-15')) AS INTEGER);
SELECT (strftime('%s','2024-03-15 12:00:00') - strftime('%s','2024-03-15 08:30:00')) / 60.0;
SELECT date('2024-03-15', '+' || 3 || ' days'), date('2024-03-15', printf('%+d months', -14));
SELECT datetime('2024-03-15 10:00:00', '+1.5 hours'), datetime('2024-03-15', '+0.5 days');
"""),
("date_ceiling_floor", """
SELECT date('2024-03-15','ceiling','+1 month'), date('2024-01-31','+1 month','floor'), date('2024-01-31','+1 month','ceiling'), date('2024-03-31','-1 month','floor');
SELECT datetime('2024-03-15 10:20:30','start of day'), datetime('2024-03-15 10:20:30','start of month'), datetime('2024-03-15 10:20:30','start of year'), time('2024-03-15 10:20:30','+1 minute');
"""),
("date_invalid_modifiers", """
SELECT date('2024-03-15','+1 fortnight'), date('2024-03-15','nonsense'), date('2024-03-15','+x days'), date('2024-03-15', NULL), date('2024-03-15','weekday 9');
SELECT date('now') IS NOT NULL, length(datetime('now')), strftime('%Y','now') >= '2024', julianday('now') > 2460000, current_date = date('now');
"""),
]

SCRIPTS["math"] = [
("math_functions", """
SELECT abs(-5), abs(-5.5), abs(NULL), abs('x'), sign(-3), sign(0), sign(2.5), round(2.5), round(-2.5), round(3.14159,2), round(1234.5678,-2), round(0.5), round(1.5);
SELECT ceil(1.2), ceiling(-1.2), floor(1.8), floor(-1.2), trunc(2.7), trunc(-2.7), mod(10,3), mod(-10,3), mod(10.5,3), mod(5,0);
"""),
("math_trig", """
SELECT round(sin(0),6), round(cos(0),6), round(tan(0.5),6), round(asin(1),6), round(acos(0),6), round(atan(1),6), round(atan2(1,1),6), round(sinh(1),6), round(cosh(1),6), round(tanh(1),6), round(asinh(1),6), round(acosh(2),6), round(atanh(0.5),6);
SELECT round(degrees(pi()),6), round(radians(180),6), pi(), round(sin(pi()),10), acos(2), asin(-2), atanh(1);
"""),
("math_exp_log", """
SELECT round(exp(1),6), round(ln(10),6), round(log(100),6), round(log10(1000),6), round(log2(8),6), round(log(2,8),6), round(log(10,1000),6), log(0), log(-1), ln(0);
SELECT round(sqrt(2),6), sqrt(-1), pow(2,10), power(2,0.5), pow(2,-1), pow(0,0), pow(-8,1.0/3), exp(1000), exp(-1000);
"""),
("integer_float_edges", """
SELECT 9223372036854775807, -9223372036854775808, 9223372036854775808, 1e308*10, -1e308*10, 5e-324, 5e-325, 0.1+0.2, 0.1+0.2=0.3;
SELECT cast(9.99 AS INTEGER), cast(-9.99 AS INTEGER), cast(1e30 AS INTEGER), cast('12abc' AS INTEGER), cast('abc' AS INTEGER), cast(' 12 ' AS INTEGER), cast('0x10' AS INTEGER), cast(x'31' AS INTEGER);
SELECT 7 / 2, 7 / 2.0, 7 % 2, -7 % 2, 7 % -2, 7.5 % 2, 1 << 63, 1 << 64, -1 >> 1, 5 / 0, 5.0 / 0;
"""),
("float_formatting", """
SELECT 1.0, 1.5, 100.0, 1e10, 1e15, 1e16, 1e21, 0.000001, 1e-7, 123456789.123456789, 1.0e0/3, 2.0/3, 100.0/3;
SELECT 3.0e0, -0.0, 0.0*-1, 1e100, 12345678901234567890.0, 0.1, 0.30000000000000004, 1/3.0*3;
SELECT cast(0.1 AS TEXT), cast(1e100 AS TEXT), cast(-1.5e-9 AS TEXT), cast(100 AS REAL), cast(1.0 AS TEXT);
"""),
("random_functions_shape", """
SELECT typeof(random()), length(randomblob(16)), typeof(randomblob(4)), abs(random()) >= 0;
SELECT count(DISTINCT random()) > 1 FROM (SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3);
SELECT length(hex(randomblob(8))), length(randomblob(0)), typeof(randomblob(0));
"""),
]

SCRIPTS["affinity"] = [
("affinity_columns", """
CREATE TABLE t(i INTEGER, r REAL, t TEXT, n NUMERIC, b BLOB, x);
INSERT INTO t VALUES('12','12','12','12','12','12');
INSERT INTO t VALUES(12.0, 12, 12, 12.0, 12, 12.0);
INSERT INTO t VALUES('abc','abc','abc','abc','abc','abc');
INSERT INTO t VALUES('1e3','1e3','1e3','1e3','1e3','1e3');
INSERT INTO t VALUES(' 5','0x10','1.50','007','','');
SELECT typeof(i),typeof(r),typeof(t),typeof(n),typeof(b),typeof(x) FROM t;
SELECT * FROM t;
"""),
("affinity_by_declared_type", """
CREATE TABLE t(a INT, b VARCHAR(10), c DOUBLE, d DECIMAL(5,2), e BOOLEAN, f DATETIME, g CLOB, h FLOATING POINT, i UNSIGNED BIG INT, j NCHAR(5));
INSERT INTO t VALUES('1','1',1,'1.5',1,'2024','x','2.5','7',9);
SELECT typeof(a),typeof(b),typeof(c),typeof(d),typeof(e),typeof(f),typeof(g),typeof(h),typeof(i),typeof(j) FROM t;
"""),
("comparison_affinity", """
CREATE TABLE t(i INTEGER, t TEXT, x);
INSERT INTO t VALUES(5,'5',5);
SELECT i=5, i='5', t=5, t='5', x=5, x='5', i=t, i<'6', t<6, x<'6' FROM t;
SELECT 1='1', 1.0='1', '1'=1, 'a'<1, 1<'a', NULL<1, x'00'>'z', x'00'>1 FROM t;
SELECT 2 < 10, '2' < '10', 2 < '10', '2' < 10, CAST(2 AS TEXT) < CAST(10 AS TEXT);
"""),
("cast_expressions", """
SELECT CAST('123' AS INTEGER), CAST(123 AS TEXT), CAST('1.5' AS REAL), CAST('1.5' AS INTEGER), CAST(1.9 AS INTEGER), CAST('abc' AS REAL), CAST(NULL AS TEXT), CAST(x'414243' AS TEXT), CAST('ABC' AS BLOB);
SELECT CAST('9223372036854775807' AS INTEGER), CAST('9223372036854775808' AS INTEGER), CAST('-9223372036854775809' AS INTEGER), CAST(1e19 AS INTEGER), CAST(-1e19 AS INTEGER);
SELECT CAST('5' AS NUMERIC), CAST('5.0' AS NUMERIC), CAST('5.5' AS NUMERIC), CAST('abc' AS NUMERIC), typeof(CAST('5.0' AS NUMERIC)), CAST(' 3 ' AS NUMERIC);
"""),
("type_ordering", """
CREATE TABLE t(v);
INSERT INTO t VALUES(NULL),(1),(2.5),('a'),('B'),(x'00'),(x'ff'),(-5),('10'),(1.0);
SELECT typeof(v), v FROM t ORDER BY v;
SELECT typeof(v), count(*) FROM t GROUP BY typeof(v) ORDER BY 1;
"""),
("numeric_text_conversion", """
SELECT '5' + 3, '5.5' + 1, 'abc' + 1, '5abc' + 1, '' + 1, ' 7 ' + 0, '0x1F' + 0, '1e2' + 0, '.5' + 0, '5.' + 0, '+5' + 0, '--5' + 0;
SELECT typeof('5' + 3), typeof('5.0' + 3), typeof('5' * '2'), typeof(5 || 5), typeof(1 + 1.0), typeof(10 / 2), typeof(10 / 4), typeof(10 / 4.0);
SELECT 9007199254740993, 9007199254740993 + 0.0, '9007199254740993' + 0, '9007199254740993.0' + 0;
"""),
("integer_primary_key_affinity", """
CREATE TABLE t(id INTEGER PRIMARY KEY, v);
INSERT INTO t VALUES(1,'a'),('2','b'),(3.0,'c');
INSERT INTO t VALUES(4.5,'d');
INSERT INTO t VALUES('x','e');
INSERT INTO t VALUES(NULL,'f');
SELECT id, typeof(id), v FROM t;
"""),
("blob_text_mix", """
CREATE TABLE t(a BLOB, b TEXT);
INSERT INTO t VALUES(x'48656c6c6f','Hello'),('Hello',x'48656c6c6f'),(1,1);
SELECT typeof(a), typeof(b), a, b, hex(a), hex(b), a=b, length(a), length(b) FROM t;
"""),
("collation_affinity", """
CREATE TABLE t(a TEXT COLLATE NOCASE, b TEXT, c TEXT COLLATE RTRIM);
INSERT INTO t VALUES('Hello','Hello','Hello  ');
SELECT a='HELLO', b='HELLO', c='Hello', a COLLATE BINARY='HELLO', c='Hello ' FROM t;
CREATE INDEX ia ON t(a);
SELECT count(*) FROM t WHERE a='hELLO';
SELECT 'a' = 'A' COLLATE NOCASE, 'a' = 'A', 'b' > 'A' COLLATE NOCASE, 'abc  ' = 'abc' COLLATE RTRIM;
"""),
("default_expressions", """
CREATE TABLE t(a DEFAULT 1, b DEFAULT 'x', c DEFAULT (2*3), d DEFAULT NULL, e DEFAULT -5, f DEFAULT 1.5, g DEFAULT TRUE, h DEFAULT CURRENT_TIMESTAMP);
INSERT INTO t(a) VALUES(NULL);
INSERT INTO t DEFAULT VALUES;
SELECT a,b,c,d,e,f,g, length(h) FROM t;
"""),
]

SCRIPTS["txn"] = [
("txn_commit_rollback", """
CREATE TABLE t(a);
BEGIN; INSERT INTO t VALUES(1); INSERT INTO t VALUES(2); COMMIT;
BEGIN; INSERT INTO t VALUES(3); ROLLBACK;
SELECT * FROM t;
BEGIN IMMEDIATE; DELETE FROM t; ROLLBACK;
SELECT count(*) FROM t;
"""),
("savepoints", """
CREATE TABLE t(a);
SAVEPOINT s1; INSERT INTO t VALUES(1);
SAVEPOINT s2; INSERT INTO t VALUES(2);
SAVEPOINT s3; INSERT INTO t VALUES(3);
ROLLBACK TO s3; SELECT * FROM t;
RELEASE s2; SELECT * FROM t;
ROLLBACK TO s1; SELECT * FROM t;
RELEASE s1; SELECT count(*) FROM t;
"""),
("txn_errors", """
CREATE TABLE t(a UNIQUE);
BEGIN;
BEGIN;
INSERT INTO t VALUES(1);
INSERT INTO t VALUES(1);
COMMIT;
COMMIT;
ROLLBACK;
SELECT * FROM t;
"""),
("statement_rollback_in_txn", """
CREATE TABLE t(a UNIQUE);
BEGIN;
INSERT INTO t VALUES(1);
INSERT INTO t VALUES(2),(3),(1),(4);
INSERT INTO t VALUES(5);
COMMIT;
SELECT * FROM t ORDER BY a;
"""),
("attach_memory", """
ATTACH DATABASE ':memory:' AS aux;
CREATE TABLE aux.t(a); CREATE TABLE main.t(a);
INSERT INTO aux.t VALUES(1),(2); INSERT INTO main.t VALUES(10);
SELECT * FROM aux.t UNION ALL SELECT * FROM main.t ORDER BY 1;
.databases
DETACH aux;
SELECT name FROM sqlite_master;
"""),
("file_database_roundtrip", """
.open test_roundtrip.db
CREATE TABLE t(a,b);
INSERT INTO t VALUES(1,'one'),(2,'two');
.open test_roundtrip.db
SELECT * FROM t;
PRAGMA integrity_check;
.open :memory:
.shell rm -f test_roundtrip.db
"""),
("vacuum_into_backup", """
CREATE TABLE t(a); INSERT INTO t SELECT value FROM json_each('[1,2,3,4,5]');
DELETE FROM t WHERE a%2=0;
VACUUM;
VACUUM INTO 'vac_copy.db';
ATTACH 'vac_copy.db' AS c;
SELECT * FROM c.t;
DETACH c;
.shell rm -f vac_copy.db
"""),
]

SCRIPTS["pragma"] = [
("pragma_settings", """
PRAGMA page_size; PRAGMA encoding; PRAGMA journal_mode; PRAGMA synchronous; PRAGMA foreign_keys; PRAGMA user_version;
PRAGMA user_version=42; PRAGMA user_version;
PRAGMA application_id=7; PRAGMA application_id;
PRAGMA cache_size; PRAGMA auto_vacuum; PRAGMA temp_store;
"""),
("pragma_schema_queries", """
CREATE TABLE a(x INTEGER PRIMARY KEY, y TEXT UNIQUE);
CREATE TABLE b(z REFERENCES a(x));
CREATE VIEW v AS SELECT * FROM a;
PRAGMA table_list;
PRAGMA foreign_key_list(b);
SELECT name, type FROM pragma_table_info('a');
SELECT * FROM pragma_index_list('a');
SELECT schema_version > 0 FROM pragma_schema_version;
"""),
("pragma_integrity_quick", """
CREATE TABLE t(a PRIMARY KEY, b); CREATE INDEX ib ON t(b);
INSERT INTO t SELECT value, value*2 FROM json_each('[1,2,3,4,5,6,7,8]');
PRAGMA integrity_check;
PRAGMA quick_check;
SELECT page_count > 0 FROM pragma_page_count;
PRAGMA freelist_count;
PRAGMA optimize;
ANALYZE;
SELECT tbl, idx, stat FROM sqlite_stat1 ORDER BY 1,2;
"""),
("pragma_compile_options_functions", """
SELECT count(*) > 5 FROM pragma_compile_options;
SELECT name FROM pragma_function_list WHERE name IN ('json_extract','sin','group_concat','printf','strftime','unixepoch') GROUP BY name ORDER BY name;
SELECT name FROM pragma_module_list WHERE name IN ('fts5','rtree','json_each','json_tree') ORDER BY name;
SELECT name FROM pragma_collation_list ORDER BY name;
"""),
]

SCRIPTS["blob"] = [
("blob_literals", """
SELECT x'48656C6C6F', hex(x'00ff'), length(x'0102030405'), typeof(x''), x'' = '', X'ab' = x'AB';
SELECT hex(zeroblob(4)), quote(x'deadbeef'), substr(x'0102030405', 2, 3), hex(substr(x'0102030405', 2, 3)), x'01' || x'02', hex(x'01' || x'02');
"""),
("blob_functions", """
SELECT hex('abc'), unhex('616263'), unhex('zz'), unhex('6'), unhex('61 62', ' '), hex(unhex('00ff')), hex(char(0, 255)), typeof(unhex('61'));
SELECT instr(x'00112233', x'1122'), length(x'00'), hex(randomblob(3)) GLOB '[0-9A-F][0-9A-F][0-9A-F][0-9A-F][0-9A-F][0-9A-F]';
SELECT hex(CAST('é' AS BLOB)), CAST(x'c3a9' AS TEXT), length(CAST(x'c3a9' AS TEXT)), length(x'c3a9');
"""),
("blob_storage_compare", """
CREATE TABLE t(id INTEGER PRIMARY KEY, b BLOB);
INSERT INTO t(b) VALUES(x'01'),(x'0100'),(x'00ff'),(zeroblob(3)),(x'');
SELECT id, hex(b), length(b) FROM t ORDER BY b, id;
SELECT max(hex(b)), min(b) IS NOT NULL FROM t;
"""),
("blob_large", """
CREATE TABLE t(b BLOB);
INSERT INTO t VALUES(zeroblob(100000));
INSERT INTO t VALUES(zeroblob(5));
SELECT length(b), hex(substr(b, 99999, 2)) FROM t ORDER BY length(b);
UPDATE t SET b = b || x'ff' WHERE length(b) = 5;
SELECT hex(b) FROM t WHERE length(b) < 10;
"""),
("blob_bitwise_checksum", """
WITH RECURSIVE n(i, h) AS (SELECT 0, 5381 UNION ALL SELECT i+1, ((h << 5) + h + (i % 251)) & 4294967295 FROM n WHERE i < 5000)
SELECT h FROM n WHERE i = 5000;
SELECT hex(CAST(x'00' || printf('%c', 200) AS BLOB));
SELECT sum(unicode(substr('hello world', value, 1))) FROM json_each('[1,2,3,4,5,6,7,8,9,10,11]');
"""),
]

SCRIPTS["collation"] = [
("collate_nocase_binary", """
CREATE TABLE t(s TEXT);
INSERT INTO t VALUES('apple'),('Banana'),('cherry'),('Apple'),('banana'),('Cherry');
SELECT s FROM t ORDER BY s;
SELECT s FROM t ORDER BY s COLLATE NOCASE, s;
SELECT DISTINCT s COLLATE NOCASE FROM t ORDER BY 1;
SELECT s, count(*) FROM t GROUP BY s COLLATE NOCASE ORDER BY 1;
"""),
("collate_index", """
CREATE TABLE t(a TEXT COLLATE NOCASE UNIQUE, b);
INSERT INTO t VALUES('x',1),('Y',2);
INSERT INTO t VALUES('X',3);
SELECT * FROM t WHERE a = 'y';
SELECT * FROM t ORDER BY a DESC;
"""),
("collate_rtrim", """
SELECT 'a' = 'a  ' COLLATE RTRIM, 'a' < 'a ' COLLATE RTRIM, 'a  b' = 'a b' COLLATE RTRIM;
CREATE TABLE t(a COLLATE RTRIM); INSERT INTO t VALUES('x'),('x '),('x  '),(' x');
SELECT count(DISTINCT a) FROM t;
SELECT a, length(a) FROM t GROUP BY a ORDER BY length(a);
"""),
("collate_in_expressions", """
SELECT 'a' IN ('A') , 'a' COLLATE NOCASE IN ('A'), ('a' COLLATE NOCASE) BETWEEN 'A' AND 'B', max('a' COLLATE NOCASE, 'B'), 'Z' COLLATE NOCASE > 'a';
SELECT 'abc' LIKE 'ABC', 'abc' = 'ABC' COLLATE NOCASE, 'abc' GLOB 'ABC';
"""),
("collate_errors", """
SELECT 'a' COLLATE nosuchcollation;
CREATE TABLE t(a COLLATE nosuch);
SELECT 'a' < 'b' COLLATE BINARY COLLATE NOCASE;
"""),
]

SCRIPTS["rtree"] = [
("rtree_basic", """
CREATE VIRTUAL TABLE r USING rtree(id, minx, maxx, miny, maxy);
INSERT INTO r VALUES(1, 0,10, 0,10),(2, 5,15, 5,15),(3, 20,30, 20,30),(4, 8,9, 8,9);
SELECT id FROM r WHERE minx>=5 AND maxx<=15 ORDER BY id;
SELECT id FROM r WHERE maxx>=9 AND minx<=9 AND maxy>=9 AND miny<=9 ORDER BY id;
SELECT count(*) FROM r;
"""),
("rtree_update_delete", """
CREATE VIRTUAL TABLE r USING rtree(id, x0, x1);
INSERT INTO r SELECT value, value*10, value*10+5 FROM json_each('[1,2,3,4,5,6,7,8]');
DELETE FROM r WHERE id%2=0;
UPDATE r SET x1 = x1 + 100 WHERE id=3;
SELECT * FROM r ORDER BY id;
SELECT id FROM r WHERE x0 < 40 AND x1 > 30 ORDER BY id;
"""),
("rtree_i32_aux", """
CREATE VIRTUAL TABLE r USING rtree_i32(id, a0, a1, b0, b1);
INSERT INTO r VALUES(1, 1.9, 4.2, 10, 20),(2, -5, 0, 1, 1);
SELECT * FROM r ORDER BY id;
CREATE VIRTUAL TABLE r2 USING rtree(id, x0, x1, +name TEXT);
INSERT INTO r2 VALUES(1, 0, 1, 'unit'),(2, 5, 6, 'far');
SELECT id, name FROM r2 WHERE x0 < 2 AND x1 > 0.5;
"""),
]

SCRIPTS["errors"] = [
("syntax_errors", """
SELECT 1;
SELEC 2;
SELECT 3;
SELECT FROM;
SELECT 4 +;
CREATE TABLE (a);
SELECT 5;
"""),
("runtime_errors", """
CREATE TABLE t(a);
SELECT nosuchcol FROM t;
SELECT * FROM nosuchtable;
SELECT nosuchfunc(1);
SELECT abs();
SELECT abs(1,2);
INSERT INTO t VALUES(1,2);
SELECT 'ok';
"""),
("constraint_errors", """
CREATE TABLE t(a PRIMARY KEY, b NOT NULL, c CHECK(c<10), d UNIQUE);
INSERT INTO t VALUES(1,1,1,1);
INSERT INTO t VALUES(1,2,2,2);
INSERT INTO t VALUES(2,NULL,2,2);
INSERT INTO t VALUES(3,3,30,3);
INSERT INTO t VALUES(4,4,4,1);
SELECT count(*) FROM t;
"""),
("shell_dot_commands", """
.help .tables
.print hello
.print "two words" 'and more'
.echo on
SELECT 1;
.echo off
.nosuchcommand
.width 5 10
.mode column
.headers on
SELECT 'ab', 'cd';
.timer off
.parameter set @x 5
SELECT @x + 1;
.bail off
SELECT 'end';
"""),
("shell_import_dump", """
CREATE TABLE t(a INTEGER, b TEXT, c REAL, d BLOB);
INSERT INTO t VALUES(1,'x''y',1.5,x'00ff'),(NULL,'multi
line',NULL,NULL);
CREATE INDEX ti ON t(a);
CREATE VIEW v AS SELECT a FROM t;
.dump
.schema --indent
"""),
("shell_read_csv", """
.mode csv
.output test_import.csv
SELECT 'id','name','score';
SELECT 1,'ann',9.5;
SELECT 2,'b,ob',7;
SELECT 3,'cy',NULL;
.output stdout
.import test_import.csv t
.mode column
.headers on
SELECT * FROM t;
SELECT typeof(score) FROM t;
.shell rm -f test_import.csv
"""),
("interrupt_free_misc", """
SELECT changes(), total_changes(), last_insert_rowid();
SELECT unlikely(1), likely(0), likelihood(1, 0.5), 1 IS NOT DISTINCT FROM 1, NULL IS NOT DISTINCT FROM NULL;
SELECT typeof(coalesce(NULL, 1.5)), min(NULL), max(NULL, 1), sign(NULL);
SELECT iif(NULL, 1, 2), iif(1, 'y'), nullif('a', 'A' COLLATE NOCASE), ifnull(NULL, NULL) IS NULL;
"""),
("big_cross_aggregates", """
CREATE TABLE n(i);
WITH RECURSIVE s(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM s WHERE i<300) INSERT INTO n SELECT i FROM s;
SELECT count(*), sum(a.i*b.i) FROM n a, n b WHERE a.i<=b.i AND (a.i+b.i)%7=0;
SELECT i%10 AS k, count(*), sum(i), min(i), max(i) FROM n GROUP BY k ORDER BY k;
CREATE INDEX ni ON n(i%10);
SELECT count(*) FROM n WHERE i%10=3;
"""),
("sorting_large", """
CREATE TABLE t(a, b);
WITH RECURSIVE s(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM s WHERE i<2000) INSERT INTO t SELECT i, (i*7919)%2003 FROM s;
SELECT a, b FROM t ORDER BY b, a LIMIT 5;
SELECT a, b FROM t ORDER BY b DESC, a DESC LIMIT 5;
SELECT count(DISTINCT b), min(b), max(b), sum(b) FROM t;
SELECT b, count(*) FROM t GROUP BY b HAVING count(*) > 1 ORDER BY 1 LIMIT 5;
"""),
]
