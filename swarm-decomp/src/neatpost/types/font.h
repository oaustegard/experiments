#pragma once
#include "fglyph.h"
/* a loaded font (font_* TU); size 0x8b0. Distinct from struct pfont (pdf TU). */
struct font {
	char pad_000[0x40];
	char desc[0x400];	/* 0x040 */
	char name[0x40];	/* 0x440 */
	char path[0x400];	/* 0x480 */
	int scale;		/* 0x880 */
	int pad_884;
	struct fglyph *glyphs;	/* 0x888 */
	int nglyphs;		/* 0x890 */
	int cap;		/* 0x894: allocated glyph slots */
	void *dict898;		/* 0x898 */
	void *dict8a0;		/* 0x8a0 */
	void *dict8a8;		/* 0x8a8 */
};
