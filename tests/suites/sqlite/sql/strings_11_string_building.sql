WITH RECURSIVE r(n,s) AS (SELECT 1,'a' UNION ALL SELECT n+1, s||char(97+n) FROM r WHERE n<10) SELECT group_concat(s, ',') FROM r;
SELECT length(zeroblob(1000)), length(printf('%1000s','x')), length(substr(printf('%5000s','y'),4000));
SELECT trim(printf('%20s','pad')), ltrim(printf('%-20s|','pad'),'p');
