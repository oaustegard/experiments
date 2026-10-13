#include "cpp_protos.h"

static long evalor(void)
{
	long v = evalbitor();
	for (;;) {
		if (enext == -1)
			enext = eval_tok();
		if (enext == 0x262600) {
			enext = -1;
			if (v) {
				v = evalbitor() != 0;
				continue;
			}
			do
				enext = eval_tok();
			while (enext == 0x262600);
		}
		if (enext == -1)
			enext = eval_tok();
		if (enext == 0x7c7c00) {
			enext = -1;
			if (!v) {
				v = evalbitor() != 0;
				continue;
			}
			do
				enext = eval_tok();
			while (enext == 0x7c7c00);
			v = 1;
		}
		return v;
	}
}
