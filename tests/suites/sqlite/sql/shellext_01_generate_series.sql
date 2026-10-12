SELECT value FROM generate_series(1, 5);
SELECT value FROM generate_series(10, 0, -3);
SELECT sum(value), count(*) FROM generate_series(1, 1000);
SELECT value FROM generate_series(0, 1, 0.25);
SELECT a.value * b.value FROM generate_series(1,3) a, generate_series(1,3) b WHERE a.value <= b.value;
SELECT value FROM generate_series(5, 1);
SELECT group_concat(value) FROM generate_series(1, 20, 7);
