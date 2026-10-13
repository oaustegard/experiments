#include "hyph_ext.h"
#include "fn_dict_free.h"
#include "tr_args.h"

void tr_hpf(struct tr_args *a)
{
	hypats_len = 0;
	hy_n = 0;
	pref_free(hypref);
	hwword_len = 0;
	hw_n = 0;
	pref_free(hwpref);
	hcode_n = 0;
	dict_free(hcodedict);
	hwpref = pref_make();
	hypref = pref_make();
	hcodedict = dict_make(0x400, 0);
	tr_hpfa(a);
}
