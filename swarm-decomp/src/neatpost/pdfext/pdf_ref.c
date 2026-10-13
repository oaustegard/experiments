#include "sys_std.h"
#include "fn_pdf_obj.h"
#include "fn_pdf_find.h"

int pdf_ref(unsigned char *s, int end, int i)
{
	int a, b;

	if (pdf_obj(s, end, i, &a, &b) != 0)
		return -1;
	return pdf_find(s, end, a, b);
}
