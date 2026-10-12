CREATE TABLE ev(id INTEGER PRIMARY KEY, body TEXT);
INSERT INTO ev(body) SELECT json_object('type', CASE WHEN value%3=0 THEN 'a' ELSE 'b' END, 'n', value) FROM json_each('[1,2,3,4,5,6,7,8,9,10]');
CREATE INDEX evt ON ev(json_extract(body,'$.type'));
SELECT count(*) FROM ev WHERE json_extract(body,'$.type')='a';
SELECT sum(json_extract(body,'$.n')) FROM ev WHERE body->>'type'='b';
