#pragma once

/* one word of the fill buffer, 56 bytes */
struct fmtword {
	char *s;		/* 0x00 */
	int wid;		/* 0x08 */
	int f0c;		/* 0x0c: wb a38[0] */
	int f10;		/* 0x10: wb a3c */
	int f14;		/* 0x14: spacing before the word */
	int f18;		/* 0x18: hyphenation width or flag */
	int f1c;		/* 0x1c */
	int f20;		/* 0x20 */
	int swid;		/* 0x24: wb_swid of the word */
	int pos;		/* 0x28 */
	int f2c;		/* 0x2c */
	int f30;		/* 0x30 */
	int f34;		/* 0x34 */
};
