VALUES(1,'a'),(2,'b'),(3,'c');
SELECT * FROM (SELECT column1 AS x, column2 AS y FROM (VALUES(1,2),(3,4))) WHERE x>1;
SELECT column1+column2 FROM (VALUES(1,2),(3,4));
