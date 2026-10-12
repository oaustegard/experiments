SELECT changes(), total_changes(), last_insert_rowid();
SELECT unlikely(1), likely(0), likelihood(1, 0.5), 1 IS NOT DISTINCT FROM 1, NULL IS NOT DISTINCT FROM NULL;
SELECT typeof(coalesce(NULL, 1.5)), min(NULL), max(NULL, 1), sign(NULL);
SELECT iif(NULL, 1, 2), iif(1, 'y'), nullif('a', 'A' COLLATE NOCASE), ifnull(NULL, NULL) IS NULL;
