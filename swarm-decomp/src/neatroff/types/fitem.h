#pragma once

/* 24-byte entry of a font's pointer table; the first word is freed by font_close */
struct fitem {
	void *ptr;	/* 0x00 */
	long a8;	/* 0x08 */
	long a16;	/* 0x10 */
};
