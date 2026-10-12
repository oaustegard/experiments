CREATE TABLE t(a); INSERT INTO t SELECT value FROM json_each('[1,2,3,4,5]');
DELETE FROM t WHERE a%2=0;
VACUUM;
VACUUM INTO 'vac_copy.db';
ATTACH 'vac_copy.db' AS c;
SELECT * FROM c.t;
DETACH c;
.shell rm -f vac_copy.db
