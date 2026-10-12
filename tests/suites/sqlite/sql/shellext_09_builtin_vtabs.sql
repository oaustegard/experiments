CREATE TABLE t(a PRIMARY KEY, b); CREATE INDEX tb ON t(b); INSERT INTO t SELECT value, value*2 FROM generate_series(1, 200);
SELECT name, path, pageno IS NOT NULL, ncell, payload FROM dbstat WHERE name = 't' ORDER BY path LIMIT 5;
SELECT name, sum(ncell), sum(payload) > 0 FROM dbstat GROUP BY name ORDER BY name;
