#include "cpp_protos.h"

static long evalcmp(void)
{
	long v = evalshift();
	for (;;) {
		if (enext == -1)
			enext = eval_tok();
		if (enext == '<') {
			enext = -1;
			v = v < evalshift();
			continue;
		}
		if (enext == -1)
			enext = eval_tok();
		if (enext == '>') {
			enext = -1;
			v = v > evalshift();
			continue;
		}
		if (enext == -1)
			enext = eval_tok();
		if (enext == 0x3c3d00) {
			enext = -1;
			v = v <= evalshift();
			continue;
		}
		if (enext == -1)
			enext = eval_tok();
		if (enext == 0x3e3d00) {
			enext = -1;
			v = v >= evalshift();
			continue;
		}
		return v;
	}
}
