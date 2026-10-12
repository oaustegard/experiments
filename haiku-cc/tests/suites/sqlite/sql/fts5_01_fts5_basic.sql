CREATE VIRTUAL TABLE docs USING fts5(title, body);
INSERT INTO docs VALUES('cats','the quick brown fox jumps over the lazy dog'),
 ('dogs','a lazy dog sleeps all day'),('foxes','the red fox and the grey fox'),('misc','nothing to see here');
SELECT title FROM docs WHERE docs MATCH 'fox' ORDER BY rank;
SELECT title FROM docs WHERE docs MATCH 'lazy AND dog' ORDER BY title;
SELECT title FROM docs WHERE body MATCH 'quick OR sleeps' ORDER BY title;
