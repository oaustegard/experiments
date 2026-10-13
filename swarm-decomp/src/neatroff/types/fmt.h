#pragma once
#include "fmtword.h"
#include "fmtline.h"

/* the fill state of one environment, 0x68 bytes */
struct fmt {
	struct fmtword *words;	/* 0x00 */
	int nwords;		/* 0x08 */
	int ncap;		/* 0x0c: allocated words */
	struct fmtline *lines;	/* 0x10 */
	int nlines;		/* 0x18 */
	int curline;		/* 0x1c */
	int linecap;		/* 0x20: allocated lines */
	char pad24[0x1c];
	int spacing;		/* 0x40 */
	int nosuppress;		/* 0x44 */
	int f48;		/* 0x48 */
	int f4c;		/* 0x4c */
	int f50;		/* 0x50 */
	int f54;		/* 0x54 */
	int f58;		/* 0x58 */
	int f5c;		/* 0x5c */
	int fillreq;		/* 0x60 */
	int f64;		/* 0x64 */
};
