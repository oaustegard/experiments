#include "sys_std.h"
#include "fn_pdf_type.h"
#include "fn_pdf_len.h"

int pdf_obj(unsigned char *s, int end, int i, int *a, int *b)
{
	if (pdf_type(s, end, i) != 'r')
		return -1;
	*a = strtol(s + i, NULL, 10);
	i += pdf_len(s, end, i);
	*b = strtol(s + i, NULL, 10);
	return 0;
}
