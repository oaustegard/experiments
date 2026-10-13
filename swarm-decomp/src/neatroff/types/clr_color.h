#pragma once

/* one entry of the named-color table: 16 bytes, rgb at 0x08 */
struct clr_color {
	const char *name;	/* 0x00 */
	int rgb;		/* 0x08: 0xRRGGBB */
};
