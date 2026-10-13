#pragma once

/* glyph-group state of one chaining-context subtable (GSUB type 6, format 3) */
struct otf_chain {
	int bt[16];
	int in[16];
	int la[16];
	int nb, ni, nl, seq;
};
