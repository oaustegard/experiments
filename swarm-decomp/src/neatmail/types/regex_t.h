#pragma once

/* offsets from regfree / regcomp / rnode_emit listings */
struct rslot {
	int c00;	/* 0x00: atom code (int store in rnode_emitnorep) */
	int pad04;
	char *str;	/* 0x08 */
	int flag;	/* 0x10: 'f' 'j' 'm' '(' ... 0 = owned literal */
	int link;	/* 0x14: patch-list link / target index */
	int c18;	/* 0x18 */
};

struct rprog {
	struct rslot *slots;	/* 0x00 */
	int n;			/* 0x08 */
	int cflags;		/* 0x0c */
};

typedef struct {
	struct rprog *prog;	/* 0x00 */
} regex_t;
