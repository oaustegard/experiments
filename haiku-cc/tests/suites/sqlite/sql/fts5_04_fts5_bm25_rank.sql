CREATE VIRTUAL TABLE d USING fts5(t);
INSERT INTO d VALUES('apple'),('apple apple banana'),('banana cherry'),('apple banana cherry apple apple');
SELECT rowid, round(bm25(d),4) FROM d WHERE d MATCH 'apple' ORDER BY bm25(d), rowid;
SELECT rowid FROM d WHERE d MATCH 'apple' ORDER BY rank, rowid;
