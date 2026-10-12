CREATE VIRTUAL TABLE d USING fts5(t);
INSERT INTO d VALUES('one two three four five'),('five four three two one'),('one three five'),('two one');
SELECT rowid FROM d WHERE d MATCH '"one two"' ORDER BY rowid;
SELECT rowid FROM d WHERE d MATCH 'NEAR(one five, 3)' ORDER BY rowid;
SELECT rowid FROM d WHERE d MATCH 'one NOT two' ORDER BY rowid;
SELECT rowid FROM d WHERE d MATCH 'th*' ORDER BY rowid;
