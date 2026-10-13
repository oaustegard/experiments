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
		break;
	}
	for (;;) {
		if (enext == -1)
			enext = eval_tok();
		if (enext == '^') {
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
				break;
			}
			v ^= r;
			continue;
		}
		break;
	}
	for (;;) {
		if (enext == -1)
			enext = eval_tok();
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
				break;
			}
			for (;;) {
				if (enext == -1)
					enext = eval_tok();
				if (enext == '^') {
					enext = -1;
					s = evaleq();
					for (;;) {
						if (enext == -1)
							enext = eval_tok();
						if (enext == '&') {
							enext = -1;
							s &= evaleq();
							continue;
						}
						break;
					}
					r ^= s;
					continue;
				}
				break;
			}
			v |= r;
			continue;
		}
		break;
	}
	return v;
}
