#include "i_jmp.h"
#include "i_rel.h"
#include "op_rr.h"
#include "op_x.h"

static void i_div(int op, int c);
static void mov_r2r(int a, int b, int t);

int i_ins(long op, long a, long b, long c, long d)
{
	static const int ucond[6] = { 0x92, 0x93, 0x94, 0x95, 0x96, 0x97 };
	static const int scond[6] = { 0x9c, 0x9d, 0x94, 0x95, 0x9e, 0x9f };
	long k = op & 0xffffff;
	long t = op & 0x20;
	char buf[4];
	char b3[3];
	int sh;
	int x;
	long off;

	if (op & 0x10) {
		if (op & 0x100000) {
			if ((unsigned long) c + 0x80 <= 0xff && b == a) {
				static const int rx[5] = { 0xc0, 0xe8, 0xe0, 0xc8, 0xf0 };

				buf[0] = 0x48 | ((b >> 3) & 1);
				buf[1] = 0x83;
				buf[2] = (b & 7) | rx[op & 0xf];
				buf[3] = c;
				mem_put(&cs, buf, 4);
			} else {
				op_rm(0x8d, a, b, c, 8);
			}
			goto L80;
		}
		goto L260;
	}
	if (!t)
		goto L89;
	if (!(op & 0x100000))
		goto L33c;
	goto L1d5;

L260:
	{
		static const int rx[5] = { 0x03, 0x2b, 0x23, 0x0b, 0x33 };

		op_rr(rx[op & 0xf], b, c, 8);
	}
	if (t)
		goto L33c;
	if (op & 0x40)
		goto L233;
	if (op & 0x80)
		goto L2c3;
	goto L9b;

L33c:
	{
		int reg = 4;

		if ((op & 0xf) == 1)
			reg = (op & 0x10000000) ? 7 : 5;
		op_rr(0xd3, reg, b, 8);
	}
	if (op & 0x40)
		goto L233;
	if (!(op & 0x80))
		goto L9b;
	goto L2c3;

L1d5:
	sh = -32;
	if (op & 1)
		sh = (op & 0x10000000) ? -8 : -24;
	buf[0] = 0x48 | ((b >> 3) & 1);
	buf[1] = 0xc1;
	buf[2] = (b & 7) | sh;
	buf[3] = c;
	mem_put(&cs, buf, 4);
	goto L89;

L80:
	if (t)
		goto L1d5;
L89:
	if (op & 0x40)
		goto L233;
L92:
	if (op & 0x80)
		goto L3d8;
L9b:
	if (op & 0x100)
		goto L430;
	if (k == 0x200)
		goto L5a8;
	if (k == 0x400200)
		goto L620;
	if (k == 0x400400)
		goto L550;
	if (k == 0x100400)
		goto L6b0;
	if (k == 0x800)
		goto L740;
	if (k == 0x801)
		goto L760;
	if (k == 0x8000)
		goto L780;
	if (k == 0x110000)
		goto L660;
	x = (int) (op >> 24);
	if (k == 0x120000)
		goto L7b8;
	if (k == 0x400)
		goto L7d2;
	x = 1;
	if (!(op & 0x7000))
		return x;
	if (op & 0x2000)
		goto L874;
L140:
	if (!(op & 0x4000))
		goto L188;
	if (!(op & 0x100000))
		goto L8bf;
	buf[0] = 0x48 | ((b >> 3) & 1);
	buf[1] = 0x83;
	buf[2] = (b & 7) | 0xf8;
	buf[3] = c;
	mem_put(&cs, buf, 4);
L188:
	off = i_jmp(op, 4);
	jmp_add(op, off, d + 1);
	return 0;

L233:
	if (k == 0x40)
		goto L4c0;
	if ((unsigned long) (k - 0x41) > 1)
		return 0;
	i_div((int) op, (int) c);
	return 0;

L2c3:
	op_rr(0x3b, b, c, 8);
L2dc:
	b3[0] = 0x0f;
	b3[1] = (op & 0x10000000) ? scond[op & 0xf] : ucond[op & 0xf];
	b3[2] = 0xc0;
	mem_put(&cs, b3, 3);
	mem_put(&cs, "H\x0f\xb6\xc0", 4);
	return 0;

L3d8:
	if (!(op & 0x100000))
		goto L2c3;
	buf[0] = 0x48 | ((b >> 3) & 1);
	buf[1] = 0x83;
	buf[2] = (b & 7) | 0xf8;
	buf[3] = c;
	mem_put(&cs, buf, 4);
	goto L2dc;

L430:
	if (k == 0x100)
		goto L600;
	if (k == 0x101)
		goto L5e8;
	if (k != 0x102)
		return 0;
	buf[0] = 0x48 | ((b >> 3) & 1);
	buf[1] = 0x83;
	buf[2] = (b & 7) | 0xf8;
	buf[3] = 0;
	mem_put(&cs, buf, 4);
	b3[0] = 0x0f;
	b3[1] = 0x94;
	b3[2] = 0xc0;
	mem_put(&cs, b3, 3);
	mem_put(&cs, "H\x0f\xb6\xc0", 4);
	return 0;

L4c0:
	if ((int) c != 2)
		op_rr(0x33, 2, 2, 4);
	op_rr(0xf7, 4, (int) c, 8);
	return 0;

L550:
	op_x(0xb8 + (a & 7), 0, a, 4);
	off = mem_len(&cs);
	i_rel(b, 0x801, off);
	op_put(c, 4);
	return 0;

L5a8:
	op_rr(0xff, 2, b, 8);
	return 0;

L5e8:
	op_rr(0xf7, 2, b, 8);
	return 0;

L600:
	op_rr(0xf7, 3, b, 8);
	return 0;

L620:
	mem_put(&cs, "\xe8", 1);
	c -= 4;
	off = mem_len(&cs);
	i_rel(b, 0x21, off);
	op_put(c, 4);
	return 0;

L660:
	{
		long t2 = op >> 24;
		int r;
		int opc = 0x8b;

		if (t2 == 4) {
			op_rm(0x8b, a, b, c, 4);
			return 0;
		}
		r = t2 & 0xf;
		if (r == 4)
			opc = (t2 & 0x10) ? 0x63 : 0x8b;
		else if (r == 2)
			opc = (t2 & 0x10) ? 0x10fbf : 0x10fb7;
		else if (r == 1)
			opc = (t2 & 0x10) ? 0x10fbe : 0x10fb6;
		op_rm(opc, a, b, c, 8);
	}
	return 0;

L6b0:
	if (b == 0) {
		op_rr(0x33, a, a, 4);
		return 0;
	}
	if ((unsigned long) b + 0xffffffffUL <= 0xfffffffeUL) {
		op_rr(0xc7, 0, a, 8);
		op_put(b, 4);
		return 0;
	}
	if ((unsigned long) (b - 1) <= 0xfffffffeUL) {
		op_x(0xb8 + (a & 7), 0, a, 4);
		op_put(b, 4);
		return 0;
	}
	op_x(0xb8 + (a & 7), 0, a, 8);
	op_put(b, 8);
	return 0;

L740:
	mem_put(&cs, "\xfc\xf3\xaa", 3);
	return 0;

L760:
	mem_put(&cs, "\xfc\xf3\xa4", 3);
	return 0;

L780:
	jmp_ret = mem_len(&cs);
	off = i_jmp(op, 4);
	jmp_add(0x1000, off, 0);
	return 0;

L7b8:
	op_rm(0x89, b, c, d, x);
	return 0;

L7d2:
	if ((x & 0xf) == 8) {
		if (a != b)
			op_rr(0x8b, a, b, 8);
		return 0;
	}
	{
		int m = x & 0xf;

		if (x & 0x10)
			m |= 0x10;
		mov_r2r(a, b, m);
	}
	return 0;

L874:
	op_rr(0x85, b, b, 8);
	goto L140;

L8bf:
	op_rr(0x3b, b, c, 8);
	goto L188;
}
