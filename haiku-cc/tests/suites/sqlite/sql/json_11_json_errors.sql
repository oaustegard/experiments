SELECT json_extract('not json', '$.a');
SELECT json_extract('{"a":1}', 'bad path');
SELECT json_array_length('{');
SELECT json_object('a');
SELECT json_set('{"a":1}', '$.a');
