#pragma once
struct glyph {
	char pad0[0x40];
	void *font;	/* 0x40 */
	short a48;	/* 0x48 */
	short a4a[2];
	short a4e;	/* 0x4e */
};
