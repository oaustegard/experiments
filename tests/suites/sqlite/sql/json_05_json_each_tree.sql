SELECT key, value, type, atom, fullkey, path FROM json_each('{"a":1,"b":[true,null],"c":{"d":"e"}}');
SELECT key, value, type FROM json_tree('{"a":[1,{"b":2}]}') ;
SELECT fullkey FROM json_tree('{"a":[1,{"b":2}]}') WHERE type NOT IN ('object','array');
