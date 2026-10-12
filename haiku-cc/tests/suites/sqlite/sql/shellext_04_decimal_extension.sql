SELECT decimal_add('1.1', '2.2'), decimal_sub('10', '0.001'), decimal_mul('123456789.123456789', '987654321.987654321'), decimal('0.1e3'), decimal('-0.0050');
SELECT decimal_cmp('1.0', '1'), decimal_cmp('2', '10'), decimal_cmp('-1', '-2'), decimal_pow2(10), decimal_exp(decimal('1e2')) IS NOT NULL;
SELECT decimal_sum(value) FROM (SELECT '0.1' AS value UNION ALL SELECT '0.2' UNION ALL SELECT '0.3');
SELECT decimal_mul(decimal_add('1e50', '1'), '3'), decimal_sub('1', '2e-30'), decimal_add('', '1'), decimal_add('abc', '1');
SELECT 'a' < 'b' COLLATE decimal, '10' < '9' COLLATE decimal, '1.50' = '1.5' COLLATE decimal;
