#include "mextend.h"

static void jmp_add(long op, long off, long dst)
{
	long sz;

	if (jmp_n == jmp_sz) {
		sz = 2 * jmp_n;
		sz = sz < 0x80 ? 0x80 : sz;
		jmp_sz = sz;
		jmp_off = mextend(jmp_off, jmp_n, sz, 8);
		jmp_dst = mextend(jmp_dst, jmp_n, jmp_sz, 8);
		jmp_op = mextend(jmp_op, jmp_n, jmp_sz, 8);
	}
	jmp_off[jmp_n] = off;
	jmp_dst[jmp_n] = dst;
	jmp_op[jmp_n] = op;
	jmp_n++;
}
