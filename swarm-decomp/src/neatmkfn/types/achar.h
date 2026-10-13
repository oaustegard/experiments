#pragma once

/* arabic-style letter table entry: base code point and positional forms (0 = none) */
struct achar {
	char *name;
	int code;	/* base code point */
	int isol;	/* isolated form */
	int init;	/* initial form */
	int medi;	/* medial form */
	int fina;	/* final form */
};
