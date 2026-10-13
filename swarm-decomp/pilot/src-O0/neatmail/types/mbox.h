#pragma once

/* Field offsets read from the listings; names are provisional. */
struct mbox {
	char pad0[0x1c0];
	int off[32];		/* 0x1c0: byte offsets, indexed by mbox_pos */
	int noff;		/* 0x240: number of entries in off[] */
	int pad1;		/* 0x244 */
	char **orig_beg;	/* 0x248 */
	char **orig_end;	/* 0x250 */
	char **cur_beg;		/* 0x258 */
	char **cur_end;		/* 0x260 */
	int pad2;		/* 0x268 */
	int nmail;		/* 0x26c */
};
