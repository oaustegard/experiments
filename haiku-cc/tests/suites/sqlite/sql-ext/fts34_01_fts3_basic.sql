CREATE VIRTUAL TABLE d USING fts3(title, body);
INSERT INTO d VALUES('One','the quick brown fox'),('Two','jumps over the lazy dog'),('Three','a fox and a dog');
SELECT rowid, title FROM d WHERE d MATCH 'fox' ORDER BY rowid;
SELECT rowid FROM d WHERE d MATCH 'fox AND dog' ORDER BY rowid;
SELECT rowid FROM d WHERE d MATCH 'fox OR lazy' ORDER BY rowid;
SELECT rowid FROM d WHERE d MATCH '"brown fox"' ORDER BY rowid;
SELECT rowid FROM d WHERE d MATCH 'title:two' ORDER BY rowid;
SELECT snippet(d, '[', ']'), offsets(d) FROM d WHERE d MATCH 'dog' ORDER BY rowid;
