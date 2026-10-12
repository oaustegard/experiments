SELECT count(*) > 5 FROM pragma_compile_options;
SELECT name FROM pragma_function_list WHERE name IN ('json_extract','sin','group_concat','printf','strftime','unixepoch') GROUP BY name ORDER BY name;
SELECT name FROM pragma_module_list WHERE name IN ('fts5','rtree','json_each','json_tree') ORDER BY name;
SELECT name FROM pragma_collation_list ORDER BY name;
