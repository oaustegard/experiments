#pragma once
#include "sbuf.h"

/* one output line of the fill buffer, 40 bytes */
struct fmtline {
	struct sbuf buf;	/* 0x00: the text of the line */
	int f10;		/* 0x10 */
	int f14;		/* 0x14 */
	int f18;		/* 0x18 */
	int f1c;		/* 0x1c */
	int f20;		/* 0x20 */
	int f24;		/* 0x24 */
};
