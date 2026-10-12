SELECT typeof(random()), length(randomblob(16)), typeof(randomblob(4)), abs(random()) >= 0;
SELECT count(DISTINCT random()) > 1 FROM (SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3);
SELECT length(hex(randomblob(8))), length(randomblob(0)), typeof(randomblob(0));
