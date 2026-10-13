#include "tr_args.h"
#include "fn_charread.h"

static void tr_eos(struct tr_args *a)
{
	char *s;

	eos_sentcnt = 0;
	eos_trancnt = 0;
	if (a->arg[0]) {
		s = a->arg[0];
		while (s) {
			if (charread(&s, eos_sent[eos_sentcnt]) < 0)
				break;
			if (eos_sentcnt <= 30)
				eos_sentcnt++;
		}
	}
	if (a->arg[1]) {
		s = a->arg[1];
		while (s) {
			if (charread(&s, eos_tran[eos_trancnt]) < 0)
				break;
			if (eos_trancnt <= 30)
				eos_trancnt++;
		}
	}
}
