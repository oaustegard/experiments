#include "ic_ent.h"
#include "ic_num.h"




int ic_num(struct ic_ent *tab, long idx, long *out)
{
	struct ic_ent *e = &tab[idx];
	long op = e->op;
	int r = 0;

	if ((op & 0x100400) == 0x100400) {
		*out = e->a;
		goto done;
	}
	if (op & 0xf0) {
		long a, b;

		if (ic_num(tab, e->a, &a)) {
			r = 1;
			goto done;
		}
		if ((r = ic_num(tab, e->b, &b)))
			goto done;
		switch (op & 0xffffff) {
		case 0x10:
			*out = b + a;
			break;
		case 0x11:
			*out = a - b;
			break;
		case 0x12:
			*out = b & a;
			break;
		case 0x13:
			*out = b | a;
			break;
		case 0x14:
			*out = b ^ a;
			break;
		case 0x20:
			*out = a << b;
			break;
		case 0x21:
			*out = (op & 0x10000000) ? a >> b : (long)((unsigned long)a >> b);
			break;
		case 0x40:
			*out = b * a;
			break;
		case 0x41:
			if (b == 0) {
				r = 1;
				goto done;
			}
			*out = a / b;
			break;
		case 0x42:
			if (b == 0) {
				r = 1;
				goto done;
			}
			*out = a % b;
			break;
		case 0x80:
			*out = b > a;
			break;
		case 0x81:
			*out = b <= a;
			break;
		case 0x82:
			*out = b == a;
			break;
		case 0x83:
			*out = b != a;
			break;
		case 0x84:
			*out = b >= a;
			break;
		case 0x85:
			*out = b < a;
			break;
		}
		goto done;
	}
	if (op & 0x100) {
		long a;
		int k;

		if (ic_num(tab, e->a, &a)) {
			r = 1;
			goto done;
		}
		k = op & 0xffffff;
		{
			long v = 0;
			if (k == 0x101)
				v = ~a;
			else if (k == 0x102)
				v = a == 0;
			else if (k == 0x100)
				v = -a;
			*out = v;
		}
		goto done;
	}
	if ((op & 0x700400) != 0x400) {
		r = 1;
		goto done;
	}
	{
		long a;
		int n;

		if (ic_num(tab, e->a, &a)) {
			r = 1;
			goto done;
		}
		n = (int)(op >> 24);
		if ((n & 0x10) == 0) {
			if ((n & 0xf) != 8) {
				n = (n & 0xf) * 8;
				a &= (1L << n) - 1;
			}
		} else if ((n & 0xf) != 8) {
			n = (n & 0xf) * 8;
			if (a > (1L << (n - 1)))
				a -= 1L << n;
		}
		*out = a;
	}
done:
	return r;
}
