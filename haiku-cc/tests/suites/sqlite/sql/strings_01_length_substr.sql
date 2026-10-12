SELECT length('hello'), length(''), length(NULL), length('héllo'), length(x'0102'), length(12345);
SELECT substr('hello',2), substr('hello',2,3), substr('hello',-3), substr('hello',-3,2), substr('hello',0), substr('hello',10), substr('hello',2,-1);
SELECT substring('hello world', 7, 5), octet_length('héllo');
