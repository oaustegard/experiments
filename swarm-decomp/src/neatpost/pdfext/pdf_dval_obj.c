#include "sys_std.h"
#include "fn_pdf_dval.h"
#include "fn_pdf_obj.h"
#include "fn_pdf_find.h"

int pdf_dval_obj(unsigned char *s, int end, int i, const char *key)
{
	int d = pdf_dval(s, end, i, key);
	int a, b;

	if (d < 0)
		return -1;
	if (pdf_obj(s, end, d, &a, &b) != 0)
		return -1;
	return pdf_find(s, end, a, b);
}
