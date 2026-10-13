#include "hyph_ext.h"
#include "fn_memset.h"

static void hcode_add(char *a, char *b)
{
	int n = dict_get(hcodedict, a);

	if (n < 0) {
		n = hcode_n;
		if (n <= 0x1ff) {
			strcpy(hcodesrc[n], a);
			strcpy(hcodedst[n], b);
			dict_put(hcodedict, hcodesrc[n], n);
			hcode_n++;
		}
	} else
		strcpy(hcodedst[n], b);
}
