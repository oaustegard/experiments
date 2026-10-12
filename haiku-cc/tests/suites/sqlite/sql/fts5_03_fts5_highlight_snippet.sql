CREATE VIRTUAL TABLE d USING fts5(t);
INSERT INTO d VALUES('The quick brown fox jumps over the lazy dog'),('Pack my box with five dozen liquor jugs');
SELECT highlight(d, 0, '[', ']') FROM d WHERE d MATCH 'fox OR box';
SELECT snippet(d, 0, '<', '>', '...', 4) FROM d WHERE d MATCH 'lazy';
