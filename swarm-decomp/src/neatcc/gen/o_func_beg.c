#include <stdlib.h>
#include "fn_o_tmpdrop.h"
#include "fn_o_back.h"
#include "fn_out_def.h"
#include "fn_mem_len.h"
#include "mextend.h"
#include "mem.h"

void o_func_beg(char *name, int argc, int global, int varg)
{
	int i;
	long n;
	long *off;
	long s;

	func_varg = varg;
	func_argc = argc;
	func_regs = 0;
	o_tmpdrop(-1);
	o_back(0);
	free(loc_off);
	loc_off = NULL;
	loc_n = 0;
	loc_sz = 0;
	loc_pos = 0;
	n = loc_n;
	off = loc_off;
	for (i = 0; i < argc; i++) {
		s = loc_sz;
		if (n >= s) {
			s = s * 2;
			if (s < 0x80)
				s = 0x80;
			loc_sz = s;
			off = mextend(off, n, s, 8);
			loc_off = off;
			n = loc_n;
		}
		off[n] = -16 - 8 * i;
		n++;
		loc_n = n;
	}
	out_def(name, global ? 17 : 1, mem_len(&cs), 0);
}
