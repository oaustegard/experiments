CREATE TABLE a(x INTEGER PRIMARY KEY, y TEXT UNIQUE);
CREATE TABLE b(z REFERENCES a(x));
CREATE VIEW v AS SELECT * FROM a;
PRAGMA table_list;
PRAGMA foreign_key_list(b);
SELECT name, type FROM pragma_table_info('a');
SELECT * FROM pragma_index_list('a');
SELECT schema_version > 0 FROM pragma_schema_version;
