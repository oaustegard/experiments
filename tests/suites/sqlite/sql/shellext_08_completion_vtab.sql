CREATE TABLE alpha(a); CREATE TABLE alphabet(b);
SELECT DISTINCT candidate FROM completion('SELECT * FROM alp', 17) ORDER BY 1;
SELECT DISTINCT candidate FROM completion('sel', 3) ORDER BY 1 LIMIT 5;
SELECT DISTINCT candidate FROM completion('', 0) ORDER BY 1 LIMIT 8;
