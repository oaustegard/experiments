SELECT cast((julianday('2024-12-25') - julianday('2024-03-15')) AS INTEGER);
SELECT (strftime('%s','2024-03-15 12:00:00') - strftime('%s','2024-03-15 08:30:00')) / 60.0;
SELECT date('2024-03-15', '+' || 3 || ' days'), date('2024-03-15', printf('%+d months', -14));
SELECT datetime('2024-03-15 10:00:00', '+1.5 hours'), datetime('2024-03-15', '+0.5 days');
