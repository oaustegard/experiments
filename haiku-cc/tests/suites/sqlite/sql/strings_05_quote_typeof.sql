SELECT quote('it''s'), quote(1), quote(1.5), quote(NULL), quote(x'ff'), quote('a'||char(10)||'b');
SELECT typeof(1), typeof(1.5), typeof('a'), typeof(x''), typeof(NULL), typeof(1+1.0), typeof('1'+1);
