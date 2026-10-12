CREATE TABLE doc(id INTEGER PRIMARY KEY, j TEXT);
INSERT INTO doc VALUES(1,'{"tags":["a","b"],"n":1}'),(2,'{"tags":["b","c","d"],"n":2}'),(3,'{"n":3}');
SELECT doc.id, e.value FROM doc, json_each(doc.j, '$.tags') e ORDER BY 1,2;
SELECT e.value, count(*) FROM doc, json_each(doc.j, '$.tags') e GROUP BY 1 ORDER BY 2 DESC, 1;
SELECT id, json_extract(j,'$.n') FROM doc WHERE json_extract(j,'$.tags[0]') = 'b';
