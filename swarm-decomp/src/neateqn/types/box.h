#pragma once
#include "sbuf.h"
struct box {
	struct sbuf buf;	/* 0x00 */
	int type;		/* 0x10 */
	int nreg_set;		/* 0x14 */
	int sreg_id;		/* 0x18 */
	int nitems;		/* 0x1c */
	int first;		/* 0x20 */
	int flags;		/* 0x24 */
	int col;		/* 0x28 */
	int pad;		/* 0x2c */
	char *wname;		/* 0x30: width register name pending a .nr */
};
