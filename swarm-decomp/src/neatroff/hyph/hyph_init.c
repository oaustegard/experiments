#include "hyph_ext.h"

void hyph_init(void)
{
	hwpref = pref_make();
	hypref = pref_make();
	hcodedict = dict_make(0x400, 0);
}
