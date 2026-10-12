SELECT json_array_length(json_group_array(value)) FROM json_each('[5,6,7]');
SELECT json_insert('[1,2]', '$[1]', 'x'), json_set('[1,2]', '$[5]', 'x'), json_remove('[1,2,3]', '$[1]');
SELECT value FROM json_each('[[1,2],[3,4]]') ;
SELECT sum(value) FROM json_tree('[[1,2],[3,[4,5]]]') WHERE type='integer';
