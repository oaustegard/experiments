CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT DEFAULT 'q' NOT NULL, c);
CREATE INDEX ic ON t(c);
PRAGMA table_info(t);
PRAGMA index_list(t);
PRAGMA index_info(ic);
