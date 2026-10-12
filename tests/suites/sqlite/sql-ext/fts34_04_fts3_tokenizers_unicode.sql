CREATE VIRTUAL TABLE s USING fts3(t, tokenize=simple);
CREATE VIRTUAL TABLE u USING fts3(t, tokenize=unicode61 'remove_diacritics=2' 'tokenchars=-');
INSERT INTO s VALUES('Hello World'),('fox-trot'),('naïve café');
INSERT INTO u VALUES('Hello World'),('fox-trot'),('naïve café');
SELECT rowid FROM s WHERE s MATCH 'hello';
SELECT rowid FROM s WHERE s MATCH 'fox';
SELECT rowid FROM u WHERE u MATCH 'fox-trot';
SELECT rowid FROM u WHERE u MATCH 'cafe';
SELECT rowid FROM u WHERE u MATCH 'naive';
