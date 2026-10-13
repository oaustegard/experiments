#pragma once

/* offsets verified against mbox_pos / mbox_savetag listings */
struct mbox {
	char *name[32];		/* 0x000 */
	long off[24];		/* 0x100 */
	int start[16];		/* 0x1c0 */
	int end[16];		/* 0x200 */
	int n;			/* 0x240 */
	int pad244;
	char **c248;		/* 0x248 */
	long *c250;		/* 0x250 */
	char **c258;		/* 0x258 */
	long *c260;		/* 0x260 */
	int pad268;
	int nc;			/* 0x26c */
};
