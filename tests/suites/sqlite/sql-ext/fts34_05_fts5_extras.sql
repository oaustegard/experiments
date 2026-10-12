CREATE VIRTUAL TABLE d USING fts5(a, b UNINDEXED, c, columnsize=0, detail=column);
INSERT INTO d VALUES('red green blue','keep','red car'),('green yellow','keep2','blue sky'),('blue blue blue','k3','nothing');
SELECT rowid FROM d WHERE d MATCH 'blue' ORDER BY rowid;
SELECT rowid FROM d WHERE d MATCH 'a:red' ORDER BY rowid;
SELECT rowid, highlight(d, 0, '<', '>') FROM d WHERE d MATCH 'blue' ORDER BY rowid;
SELECT rowid FROM d WHERE d MATCH 'blue' ORDER BY bm25(d, 10.0, 0.0, 1.0) LIMIT 2;
INSERT INTO d(d, rank) VALUES('rank', 'bm25(2.0)');
INSERT INTO d(d) VALUES('optimize');
SELECT rowid FROM d WHERE d MATCH 'red OR yellow' ORDER BY rank;
