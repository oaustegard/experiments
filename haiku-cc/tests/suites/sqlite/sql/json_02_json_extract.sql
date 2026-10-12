SELECT json_extract('{"a":1,"b":{"c":[10,20,30],"d":"s"}}', '$.a');
SELECT json_extract('{"a":1,"b":{"c":[10,20,30],"d":"s"}}', '$.b.c[1]', '$.b.d');
SELECT json_extract('{"a":1,"b":{"c":[10,20,30],"d":"s"}}', '$.b.c');
SELECT json_extract('{"a":1}', '$.zz'), json_extract('[1,2,3]', '$[#-1]');
SELECT '{"a":{"b":5}}' -> '$.a', '{"a":{"b":5}}' ->> '$.a.b', '[10,20]' ->> 1, '{"k":"v"}' -> 'k';
