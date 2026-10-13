#include "i_jmp.h"

static long i_jmp(long op, int n)
{
	static const int ucond[6] = { 0x92, 0x93, 0x94, 0x95, 0x96, 0x97 };
	static const int scond[6] = { 0x9c, 0x9d, 0x94, 0x95, 0x9e, 0x9f };
	static char buf[16];
	char b[2];
	long k;
	long m;
	int cc;
	int i;

	if (op & 0x2000) {
		k = op & 0xffffff;
		b[0] = 0x0f;
		b[1] = 0;
		cc = (k != 0x2000) - 0x7c;
		if (n == 1) {
			b[0] = (cc & 0xf) | 0x70;
			mem_put(&cs, b, 1);
		} else {
			b[1] = cc;
			mem_put(&cs, b, 2);
		}
	} else if (op & 0x4000) {
		if (op & 0x10000000)
			cc = scond[op & 0xf];
		else
			cc = ucond[op & 0xf];
		b[0] = 0x0f;
		b[1] = 0;
		cc &= ~0x10;
		if (n == 1) {
			b[0] = (cc & 0xf) | 0x70;
			mem_put(&cs, b, 1);
		} else {
			b[1] = cc;
			mem_put(&cs, b, 2);
		}
	} else {
		if (n == 1)
			mem_put(&cs, "\xeb", 1);
		else
			mem_put(&cs, "\xe9", 1);
	}
	m = n;
	for (i = 0; i < n; i++)
		buf[i] = 0;
	mem_put(&cs, buf, m);
	return mem_len(&cs) - m;
}
