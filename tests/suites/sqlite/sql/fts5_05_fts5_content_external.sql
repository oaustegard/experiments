CREATE TABLE src(id INTEGER PRIMARY KEY, txt);
CREATE VIRTUAL TABLE idx USING fts5(txt, content='src', content_rowid='id');
INSERT INTO src VALUES(1,'alpha beta'),(2,'beta gamma'),(3,'gamma delta');
INSERT INTO idx(rowid, txt) SELECT id, txt FROM src;
SELECT rowid FROM idx WHERE idx MATCH 'beta' ORDER BY rowid;
INSERT INTO idx(idx) VALUES('integrity-check');
SELECT count(*) FROM idx WHERE idx MATCH 'delta OR alpha';
