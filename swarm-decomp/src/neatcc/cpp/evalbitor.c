#include "cpp_protos.h"

static long evalbitor(void)
{
	long v = evaleq();
	long r, s;
	for (;;) {
		if (enext == -1)
			enext = eval_tok();
		if (enext == '&') {
			enext = -1;
			v &= evaleq();
			continue;
		}
		if (enext == '^') {
			enext = -1;
			r = evaleq();
			while (enext == -1 || enext == '&') {
				if (enext == -1)
					enext = eval_tok();
				if (enext != '&')
					break;
				enext = -1;
				r &= evaleq();
			}
			v ^= r;
			continue;
		}
		if (enext == '|') {
			enext = -1;
			r = evaleq();
			for (;;) {
				if (enext == -1)
					enext = eval_tok();
				if (enext == '&') {
					enext = -1;
					r &= evaleq();
					continue;
				}
				if (enext == -1)
					enext = eval_tok();
				if (enext == '^') {
					enext = -1;
					s = evaleq();
					while (enext == -1 || enext == '&') {
						if (enext == -1)
							enext = eval_tok();
						if (enext != '&')
							break;
						enext = -1;
						s &= evaleq();
					}
					r ^= s;
					continue;
				}
				break;
			}
			v |= r;
			continue;
		}
		return v;
	}
}
