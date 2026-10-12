SELECT json_object('a',1,'b','two','c',NULL,'d',json_array(1,2)), json_array(1,'x',NULL,2.5,json('{"q":1}'));
SELECT json_array(), json_object(), json_quote('a"b'), json_quote(3), json_quote(NULL);
SELECT json_array_length('[1,[2,3],4]'), json_array_length('[1,[2,3],4]','$[1]'), json_array_length('{"a":1}');
