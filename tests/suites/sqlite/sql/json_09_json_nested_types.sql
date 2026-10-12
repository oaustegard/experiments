SELECT json_extract('{"a":1.0}', '$.a'), json_extract('{"a":1e3}', '$.a'), json_extract('{"a":12345678901234567890}', '$.a');
SELECT json_extract('{"a":"\u00e9\n"}', '$.a'), json_quote('é'), json('{"a":"\u00e9"}');
SELECT typeof(json_extract('{"a":true}','$.a')), json_extract('{"a":true}','$.a'), json_extract('{"a":null}','$.a') IS NULL;
