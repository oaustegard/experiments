SELECT 'ABC' LIKE 'abc', 'ABC' GLOB 'abc', 'abc' GLOB 'a?c', 'abc' GLOB 'a[a-c]c', 'a.c' LIKE 'a.c', 'abc' LIKE 'a_c';
PRAGMA case_sensitive_like=ON;
SELECT 'ABC' LIKE 'abc', 'abc' LIKE 'abc';
SELECT like('a%','abc'), glob('a*','abc'), like('A_','ab','\');
