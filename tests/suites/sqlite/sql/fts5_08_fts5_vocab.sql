CREATE VIRTUAL TABLE d USING fts5(t);
INSERT INTO d VALUES('a b c'),('a b'),('a');
CREATE VIRTUAL TABLE v USING fts5vocab(d, 'row');
SELECT term, doc FROM v ORDER BY term;
CREATE VIRTUAL TABLE v2 USING fts5vocab(d, 'col');
SELECT term, col, doc, cnt FROM v2 ORDER BY term;
