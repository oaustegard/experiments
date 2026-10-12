CREATE TABLE emp(id INTEGER PRIMARY KEY, name TEXT, dept TEXT, salary REAL, boss INTEGER, hired TEXT);
INSERT INTO emp VALUES
 (1,'Ada','eng',120.5,NULL,'2015-03-01'),(2,'Bob','eng',100,1,'2016-07-15'),(3,'Cy','ops',80,1,'2017-01-20'),
 (4,'Di','ops',85.25,3,'2018-11-30'),(5,'Ed','eng',95,2,'2019-05-05'),(6,'Flo','hr',70,1,'2020-02-29'),
 (7,'Gus','hr',72,6,'2021-08-08'),(8,'Hal','eng',130,1,'2014-12-12'),(9,'Ivy','ops',NULL,3,'2022-04-04');

SELECT id, name, lag(name) OVER (ORDER BY id), lead(name, 2, 'none') OVER (ORDER BY id) FROM emp ORDER BY id;
SELECT id, salary, salary - lag(salary,1,0) OVER (PARTITION BY dept ORDER BY id) FROM emp ORDER BY id;
