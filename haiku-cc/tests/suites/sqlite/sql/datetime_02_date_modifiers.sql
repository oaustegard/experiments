SELECT date('2024-03-15','+1 day'), date('2024-03-15','-1 month'), date('2024-01-31','+1 month'), date('2024-02-29','+1 year'), date('2024-03-15','start of month'), date('2024-03-15','start of year');
SELECT date('2024-03-15','weekday 0'), date('2024-03-15','weekday 5'), date('2024-03-15','+10 days','start of month','+1 month','-1 day');
SELECT datetime('2024-03-15 12:00:00','+36 hours'), datetime('2024-03-15 12:00:00','-90 minutes'), time('12:00:00','+3600 seconds'), datetime('2024-03-15','start of day','+1 day','-1 second');
