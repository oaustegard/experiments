#include "fn_dict_free.h"
#include "hyph_ext.h"

void hyph_done(void)
{
	if (hwpref)
		pref_free(hwpref);
	if (hypref)
		pref_free(hypref);
	if (hcodedict)
		dict_free(hcodedict);
}
