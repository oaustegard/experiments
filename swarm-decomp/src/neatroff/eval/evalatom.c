#include <ctype.h>

static long evalatom(char **sp)
{
	char *s = *sp;
	unsigned char c;
	char d;
	long r = 0, sg = 1, n = 0, sc = 0;
	char *e;
	int u;

	for (;;) {
		c = *s;
		if (c == '.' || isdigit(c))
			break;
		if (c == '-')
			sg = -sg;
		else if (c == '+')
			;
		else if (c == '|')
			r += abspos * sg;
		else if (c == '(') {
			long v;

			*sp = s + 1;
			v = evalexpr(sp);
			if (**sp == ')')
				*sp = *sp + 1;
			return r + v * sg;
		} else
			return r;
		*sp = ++s;
	}
	d = *s++;
	for (;;) {
		if (isdigit(d)) {
			if (sc != 100000 && !(sc > 0 && n > 200000000)) {
				sc *= 10;
				n = n * 10 + (d - '0');
			}
		} else if (d == '.') {
			if (sc != 100000 && !(sc > 0 && n > 200000000))
				sc = 1;
		} else
			break;
		d = *s++;
	}
	e = s - 1;
	if (d && strchr("icpPvmnu", d)) {
		u = d;
		e = s;
	} else
		u = defunit;
	switch (u) {
	case 'i':
		n = n * dev_res;
		break;
	case 'v':
		n = n * *nreg('v');
		break;
	case 'm':
		n = n * *nreg('s') * dev_res / 72;
		break;
	case 'n':
		n = n * *nreg('s') * dev_res / 144;
		break;
	case 'p':
		n = n * dev_res / 72;
		break;
	case 'P':
		n = n * dev_res / 6;
		break;
	case 'c':
		n = n * dev_res * 50 / 127;
		break;
	}
	*sp = e;
	return r + sg * (n / (sc > 0 ? sc : 1));
}
