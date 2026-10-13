#include "cpp_protos.h"

static long evalshift(void)
{
	long v = evaladd();
	for (;;) {
		if (enext == -1)
			enext = eval_tok();
		if (enext == 0x3c3c00) {
			enext = -1;
			v <<= evaladd();
			continue;
		}
		if (enext == -1)
			enext = eval_tok();
		if (enext == 0x3e3e00) {
			enext = -1;
			v >>= evaladd();
			continue;
		}
		return v;
	}
}
