SELECT date('2000-02-29','+1 year'), date('1900-02-28','+1 day'), date('2100-02-28','+1 day'), date('2024-12-31','+1 day'), date('2024-01-01','-1 day');
SELECT date('0000-01-01'), date('9999-12-31'), date('9999-12-31','+1 day'), date('-0001-01-01'), date('0001-01-01','-1 day');
SELECT julianday('9999-12-31'), unixepoch('0000-01-01'), unixepoch('9999-12-31 23:59:59');
