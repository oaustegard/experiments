SELECT 'a'||'b', 'a'||NULL, concat('a',NULL,1,2.5), concat_ws('-','a',NULL,'b'), concat_ws(',',1,2,3);
SELECT 1||2, 1.5||'x', hex('abc'), unhex('616263'), char(72,105,128512), unicode('A'), unicode('€');
