CREATE TABLE cust(id, name); CREATE TABLE ord(id, cust, item); CREATE TABLE item(id, label, price);
INSERT INTO cust VALUES(1,'ann'),(2,'ben');
INSERT INTO item VALUES(1,'pen',1.5),(2,'ink',4),(3,'pad',2.25);
INSERT INTO ord VALUES(1,1,1),(2,1,3),(3,2,2),(4,2,2),(5,1,2);
SELECT c.name, i.label, i.price FROM cust c JOIN ord o ON o.cust=c.id JOIN item i ON i.id=o.item ORDER BY c.name, o.id;
SELECT c.name, sum(i.price) FROM cust c JOIN ord o ON o.cust=c.id JOIN item i ON i.id=o.item GROUP BY c.name ORDER BY 1;
