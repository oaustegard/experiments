#pragma once
#include "libc.h"
#include "dict.h"
#include "fglyph.h"
#include "iset.h"

/* one substitution/positioning rule entry: 16 bytes */
struct fent {
	int code;		/* 0x00: glyph index */
	short flags;		/* 0x04: bit 1 and 2 mark skipped entries */
	short aux;		/* 0x06: sscanf "%hd" fields 1..4 of a rule */
	short a8;		/* 0x08 */
	short aa;		/* 0x0a */
	short ac;		/* 0x0c */
	short ae;		/* 0x0e */
};

/* one rule list of the font: 24 bytes */
struct frule {
	struct fent *ent;	/* 0x00: entries */
	int unk8;		/* 0x08 */
	short n;		/* 0x0c: number of entries */
	short feat;		/* 0x0e: feature index into font.feat */
	short scrp;		/* 0x10: script index, must equal font.scrp unless < 0 */
	short lang;		/* 0x12: language index, must equal font.lang unless < 0 */
	int pad14;		/* 0x14 */
};

/* font descriptor as built by font_open; 0xad8 bytes */
struct font {
	char name[32];		/* 0x00: font name (fscanf "name") */
	char fontname[32];	/* 0x20: fontname (fscanf "fontname") */
	int spacewidth;		/* 0x40 */
	int special;		/* 0x44: special font flag */
	int cs[2];		/* 0x48: constant width size and flag */
	int bd;			/* 0x50: bold */
	int zoom;		/* 0x54: zoom */
	int track[4];		/* 0x58: track kerning values */
	struct fglyph *glyph;	/* 0x68: glyph table, 88 bytes per glyph */
	int nglyph;		/* 0x70 */
	int pad74;		/* 0x74 */
	struct dict *d78;	/* 0x78: name -> glyph index (mapping) */
	struct dict *d80;	/* 0x80: name -> glyph index (fallback) */
	struct dict *d88;	/* 0x88: name -> glyph index */
	char featname[128][8];	/* 0x90: feature names */
	int feat[128];		/* 0x490: feature values */
	char scrpname[64][8];	/* 0x690: script names */
	int scrp;		/* 0x890: current script index */
	char langname[64][8];	/* 0x894: language names */
	int lang;		/* 0xa94: current language index */
	struct frule *gsub;	/* 0xa98: entries, freed by font_close */
	int ngsub;		/* 0xaa0 */
	int capgsub;		/* 0xaa4: allocated gsub rules */
	struct frule *gpos;	/* 0xaa8: entries, freed by font_close */
	int ngpos;		/* 0xab0 */
	int capgpos;		/* 0xab4: allocated gpos rules */
	struct iset *iab8;	/* 0xab8 */
	struct iset *iac0;	/* 0xac0 */
	struct iset *iac8;	/* 0xac8 */
	struct iset *iad0;	/* 0xad0 */
};
