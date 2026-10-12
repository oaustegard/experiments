SELECT json_set('{"a":1}', '$.b', 2, '$.a', 10);
SELECT json_insert('{"a":1}', '$.a', 99, '$.c', 3);
SELECT json_replace('{"a":1}', '$.a', 'new', '$.zz', 1);
SELECT json_remove('{"a":1,"b":2,"c":[1,2,3]}', '$.b', '$.c[0]');
SELECT json_patch('{"a":1,"b":{"c":2}}', '{"b":{"c":null,"d":4},"e":5}');
SELECT json_set('[1,2,3]', '$[#]', 4), json_set('{}', '$.a.b', 1);
