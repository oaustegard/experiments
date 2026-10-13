#include "ren_ps.h"

static void ren_ps(char *s)
{
	int n;
	int *q;
	int *p;

	if (s == 0 || *s == 0) {
		q = nreg(map(".s0"));
		n = dev_res / 72 * *q;
	} else if (*s != '0' || s[1] != 0) {
		q = nreg('s');
		n = eval_re(s, dev_res / 72 * *q, 'p');
	} else {
		q = nreg(map(".s0"));
		n = dev_res / 72 * *q;
	}
	p = nreg('s');
	q = nreg(map(".s0"));
	*q = *p;
	n = (n + dev_res / 144) / (dev_res / 72);
	*nreg('s') = n <= 0 ? 1 : n;
}
