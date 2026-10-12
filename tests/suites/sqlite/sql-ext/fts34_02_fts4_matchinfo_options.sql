CREATE VIRTUAL TABLE d USING fts4(a, b, prefix='2,4', tokenize=porter, order=desc);
INSERT INTO d VALUES('running quickly','dogs running'),('runs and runners','a dog'),('quick brown','nothing here');
SELECT rowid FROM d WHERE d MATCH 'run*' ORDER BY rowid;
SELECT rowid, hex(matchinfo(d)) FROM d WHERE d MATCH 'dog' ORDER BY rowid;
SELECT rowid, hex(matchinfo(d, 'pcnalx')) FROM d WHERE d MATCH 'quick OR dogs' ORDER BY rowid;
SELECT docid FROM d WHERE d MATCH 'running NEAR dogs' ORDER BY docid;
INSERT INTO d(d) VALUES('optimize');
INSERT INTO d(d) VALUES('integrity-check');
