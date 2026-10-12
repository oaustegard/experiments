SELECT format('%d items', 3), format('%5.1f|%-6s|%+d', 3.14159, 'ab', 5), format('%x %X %o %c', 255, 255, 8, 65);
SELECT format('%q %Q %w', 'it''s', 'it''s', 'a"b'), format('%Q', NULL), format('%s', NULL), format('%%');
SELECT format('%10.3e', 12345.6789), format('%g %g %g', 0.0001, 123456789.0, 1e-10), format('%05d|%-5d|', 42, 42);
