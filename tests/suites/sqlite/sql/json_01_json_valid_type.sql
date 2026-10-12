SELECT json_valid('{"a":1}'), json_valid('{a:1}'), json_valid(''), json_valid('[1,2,'), json_valid('null');
SELECT json_type('{"a":[1,2.5,"x",null,true]}'), json_type('{"a":[1,2.5,"x",null,true]}', '$.a'), json_type('[1,2]', '$[1]');
SELECT json_type('[1,2.5,"x",null,true,false]','$[0]'), json_type('[1,2.5,"x",null,true,false]','$[1]'), json_type('[1,2.5,"x",null,true,false]','$[2]'), json_type('[1,2.5,"x",null,true,false]','$[3]'), json_type('[1,2.5,"x",null,true,false]','$[4]'), json_type('[1,2.5,"x",null,true,false]','$[5]');
