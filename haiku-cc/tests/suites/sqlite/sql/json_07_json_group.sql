CREATE TABLE t(g, k, v);
INSERT INTO t VALUES('x','a',1),('x','b',2),('y','c',3);
SELECT g, json_group_array(v), json_group_object(k, v) FROM t GROUP BY g ORDER BY g;
SELECT json_group_array(json_object('k',k,'v',v)) FROM t;
