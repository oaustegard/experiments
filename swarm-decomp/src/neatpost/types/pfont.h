#pragma once
/* one embedded font of the pdf TU; size 0x898 (stride of pfonts[]) */
struct pfont {
	char name[0x80];	/* 0x000: font name, snprintf "%s" */
	char path[0x400];	/* 0x080: font path */
	char desc[0x400];	/* 0x480: font description */
	int minc;		/* 0x880: lowest glyph code used */
	int maxc;		/* 0x884: highest glyph code used */
	int f888;		/* 0x888: sub-font number (0 for non-type-1) */
	int objid;		/* 0x88c: pdf object id */
	int wid;		/* 0x890: width table object */
	int f894;		/* 0x894: nonzero for 2-byte (type t) fonts */
};
