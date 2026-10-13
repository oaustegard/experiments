#pragma once

/* one entry of a font's glyph table: 88 bytes per glyph (index * 88 from font.g) */
struct fglyph {
	char b[88];
};
