SELECT strftime('%Y-%m-%d %H:%M:%S', '2024-03-15 13:45:30'), strftime('%d/%m/%y', '2024-03-05'), strftime('%j', '2024-03-15'), strftime('%w %W %U', '2024-03-15');
SELECT strftime('%s', '2024-03-15 00:00:00'), strftime('%f', '2024-03-15 13:45:30.123'), strftime('%J', '2000-01-01 12:00'), strftime('%%|%H|%I|%M|%p', '2024-03-15 15:07:00');
SELECT strftime('%G-W%V-%u', '2024-03-15'), strftime('%e|%k|%l|%P', '2024-03-05 09:05:00'), strftime('%a %A %b %B', '2024-03-15'), strftime('%F %T %R', '2024-03-15 13:45:30');
