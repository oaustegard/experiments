#include "i_jmp.h"

void i_code(char **pcode, long *plen, long **psym, long **pflg, long **poff, long *prn)
{
	static char buf[16];
	long n = jmp_n;
	int *sz = malloc(4 * n);
	char *old;
	long cl;
	long i, cur, d, start, disp;
	int k, j;

	for (i = 0; i < n; i++)
		sz[i] = 4;
	cl = mem_len(&cs);
	old = mem_get(&cs);
	for (i = 0; i < jmp_n; i++)
		sz[i] = (unsigned) (0x6f + lab_loc[jmp_dst[i]] - jmp_off[i]) < 0xdf ? 1 : 4;
	cur = 0;
	d = 0;
	j = 0;
	k = 1;
	for (i = 0; i < jmp_n; i++) {
		start = jmp_off[i] - (1 + ((jmp_op[i] & 0x6000) != 0));
		for (; j < rel_n && rel_off[j] <= start; j++)
			rel_off[j] += d;
		for (; k < lab_sz && lab_loc[k] <= start; k++)
			lab_loc[k] += d;
		mem_put(&cs, old + cur, start - cur);
		jmp_off[i] = i_jmp(jmp_op[i], sz[i]);
		cur = start + 5 + ((jmp_op[i] & 0x6000) != 0);
		d = mem_len(&cs) - cur;
	}
	for (; j < rel_n; j++)
		rel_off[j] += d;
	for (; k < lab_sz; k++)
		lab_loc[k] += d;
	lab_loc[0] += d;
	mem_put(&cs, old + cur, cl - cur);
	free(old);
	for (i = 0; i < jmp_n; i++) {
		int s = sz[i];
		long m;

		disp = lab_loc[jmp_dst[i]] - jmp_off[i] - s;
		for (m = 0; m < s; m++) {
			buf[m] = disp;
			disp >>= 8;
		}
		mem_cpy(&cs, jmp_off[i], buf, s);
	}
	free(sz);
	*plen = mem_len(&cs);
	*pcode = mem_get(&cs);
	rel_sz = 0;
	jmp_n = 0;
	*psym = rel_sym;
	rel_sym = 0;
	*pflg = rel_flg;
	rel_flg = 0;
	*poff = rel_off;
	rel_off = 0;
	*prn = rel_n;
	rel_n = 0;
}
