CREATE VIRTUAL TABLE d USING fts5(a, b);
INSERT INTO d VALUES('x y','z'),('y z','x'),('q','r');
UPDATE d SET a='changed' WHERE rowid=1;
DELETE FROM d WHERE rowid=3;
SELECT rowid, * FROM d ORDER BY rowid;
SELECT rowid FROM d WHERE d MATCH 'a:changed OR b:x' ORDER BY rowid;
SELECT rowid FROM d WHERE d MATCH '{a b}:y' ORDER BY rowid;
