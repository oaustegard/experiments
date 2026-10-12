ATTACH DATABASE ':memory:' AS aux;
CREATE TABLE aux.t(a); CREATE TABLE main.t(a);
INSERT INTO aux.t VALUES(1),(2); INSERT INTO main.t VALUES(10);
SELECT * FROM aux.t UNION ALL SELECT * FROM main.t ORDER BY 1;
.databases
DETACH aux;
SELECT name FROM sqlite_master;
