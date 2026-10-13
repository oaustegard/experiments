#include "sys_std.h"
#include "fn_pdf_dval.h"
#include "fn_pdf_type.h"
#include "fn_pdf_obj.h"
#include "fn_pdf_find.h"

int pdf_dval_val(unsigned char *s, int end, int i, const char *key)
{
	int d = pdf_dval(s, end, i, key);
	int a, b;

	if (d < 0)
		return -1;
	if (pdf_type(s, end, d) == 'r') {
		pdf_obj(s, end, d, &a, &b);
		return pdf_find(s, end, a, b);
	}
	return d;
}
