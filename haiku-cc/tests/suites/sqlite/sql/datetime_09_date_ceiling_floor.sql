SELECT date('2024-03-15','ceiling','+1 month'), date('2024-01-31','+1 month','floor'), date('2024-01-31','+1 month','ceiling'), date('2024-03-31','-1 month','floor');
SELECT datetime('2024-03-15 10:20:30','start of day'), datetime('2024-03-15 10:20:30','start of month'), datetime('2024-03-15 10:20:30','start of year'), time('2024-03-15 10:20:30','+1 minute');
