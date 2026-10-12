WITH RECURSIVE sp(w, rest) AS (SELECT '', 'the,quick,brown,fox,' UNION ALL SELECT substr(rest,1,instr(rest,',')-1), substr(rest,instr(rest,',')+1) FROM sp WHERE rest<>'')
SELECT w FROM sp WHERE w<>'';
SELECT group_concat(upper(substr(w,1,1))||substr(w,2), ' ') FROM (SELECT 'hello' w UNION ALL SELECT 'big' UNION ALL SELECT 'world');
