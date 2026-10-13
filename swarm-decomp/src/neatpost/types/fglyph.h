#pragma once
struct font;

/* one glyph record of a font; 88 (0x58) bytes. name is read with %s at offset 0 */
struct fglyph {
	char name[0x20];	/* 0x000 */
	char key[0x20];		/* 0x020: second name, copied from the font line */
	struct font *font;	/* 0x040 */
	int width;		/* 0x048 */
	int code0;		/* 0x04c: code from the "code name" line */
	int code;		/* 0x050 */
	int pad_054;
};
