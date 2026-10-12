SELECT julianday('2000-01-01'), julianday('2000-01-01 12:00:00'), julianday('2024-03-15') - julianday('2024-01-01');
SELECT unixepoch('1970-01-02'), unixepoch('2024-03-15 10:00:00'), datetime(0,'unixepoch'), datetime(1710497000,'unixepoch'), datetime(1710497000.5,'unixepoch','subsec');
SELECT date(2460385.5), datetime(2460385.5), date(2460385.5,'julianday'), datetime(1710497000,'auto'), datetime(2460385.5, 'auto');
