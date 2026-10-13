#pragma once

/* offsets from rnode_parse / rnode_count / rnode_free listings */
struct rnode {
	int c00;		/* 0x00: atom code, e.g. '^' '.' '$' '[' */
	int pad04;
	char *str;		/* 0x08 */
	struct rnode *left;	/* 0x10 */
	struct rnode *right;	/* 0x18 */
	int min;		/* 0x20 */
	int max;		/* 0x24 */
	int c28;		/* 0x28: group number */
	int type;		/* 0x2c: '|', '(' or 'c' */
};
