SELECT time('23:59:59','+1 second'), time('00:00:00','-1 second'), datetime('12:30'), date('12:30'), time('12:30:15.678'), time('12:30:15.678','subsec');
SELECT datetime('2024-03-15 12:30:00','localtime','utc') IS NOT NULL, datetime('2024-03-15 12:30:00','utc') IS NOT NULL;
