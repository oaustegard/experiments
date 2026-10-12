SELECT 'hello' REGEXP 'h.*o', 'hello' REGEXP '^e', 'a1b2' REGEXP '[0-9]+', 'ABC' REGEXP '[a-c]+', 'foo.bar' REGEXP 'o\.b';
CREATE TABLE t(s); INSERT INTO t VALUES('apple pie'),('banana'),('cherry tart'),('date');
SELECT s FROM t WHERE s REGEXP '(an|er)+' ORDER BY s;
SELECT s FROM t WHERE s REGEXP '^[a-c].* [pt]' ORDER BY s;
SELECT regexp('a+b', 'caaab'), regexp('^$', ''), regexp('\d{3}', 'ab123');
