CREATE TABLE ev(id, d);
INSERT INTO ev VALUES(1,'2024-01-15'),(2,'2024-02-20'),(3,'2024-02-29'),(4,'2023-12-31');
SELECT id, d, strftime('%Y-%m', d) ym, cast(strftime('%j', d) AS INT) doy, julianday(d)-julianday('2024-01-01') AS off FROM ev ORDER BY d;
SELECT strftime('%Y-%m', d), count(*) FROM ev GROUP BY 1 ORDER BY 1;
SELECT id FROM ev WHERE d BETWEEN date('2024-02-01') AND date('2024-02-01','+1 month','-1 day') ORDER BY id;
