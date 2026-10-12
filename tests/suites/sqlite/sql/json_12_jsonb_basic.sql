SELECT json(jsonb('{"a":[1,2,3],"b":"x"}'));
SELECT jsonb_extract('{"a":[1,2,3]}', '$.a') IS NOT NULL, json_extract(jsonb('{"a":[1,2,3]}'), '$.a[2]');
SELECT json(jsonb_set(jsonb('{"a":1}'), '$.b', 2)), json(jsonb_array(1,2,'x')), json(jsonb_object('k', 1));
SELECT hex(jsonb('{"a":1}')) <> '', typeof(jsonb('[1]'));
