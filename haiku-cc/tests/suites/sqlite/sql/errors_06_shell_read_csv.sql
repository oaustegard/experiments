.mode csv
.output test_import.csv
SELECT 'id','name','score';
SELECT 1,'ann',9.5;
SELECT 2,'b,ob',7;
SELECT 3,'cy',NULL;
.output stdout
.import test_import.csv t
.mode column
.headers on
SELECT * FROM t;
SELECT typeof(score) FROM t;
.shell rm -f test_import.csv
