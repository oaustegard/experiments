SELECT date('2024-03-15','+1 fortnight'), date('2024-03-15','nonsense'), date('2024-03-15','+x days'), date('2024-03-15', NULL), date('2024-03-15','weekday 9');
SELECT date('now') IS NOT NULL, length(datetime('now')), strftime('%Y','now') >= '2024', julianday('now') > 2460000, current_date = date('now');
