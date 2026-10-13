#include "font_readchar.h"
#include "font_readgpos.h"
#include "font_readgsub.h"
#include "font_readfeat.h"
#include "font_isetinsert.h"
#include "font_findfeat.h"
#include "font_ctype.h"

struct font *font_open(char *name)
{
	FILE *fp;
	struct font *f;
	char buf[128];
	char lig[512][32];
	struct frule *r;
	int c1 = -1, c2 = 0, sec = 0, fi = 0, sc = 0, lg = 0, b;
	int n, i, j, k, x, feat, code, nlig = 0, sel = 0;

	fp = fopen(name, "r");
	if (fp == NULL)
		return NULL;
	f = xmalloc(sizeof *f);
	memset(f, 0, sizeof *f);
	f->d78 = dict_make(0x400, 1);
	f->d80 = dict_make(0x400, 1);
	f->d88 = dict_make(0x400, 1);
	f->iac8 = iset_make();
	f->iad0 = iset_make();

	while (fscanf(fp, "%127s", buf) == 1) {
		if (strcmp(buf, "char") == 0) {
			font_readchar(f, fp, &c2, &c1);
		} else if (strcmp(buf, "kern") == 0) {
			char w1[32], w2[128];
			int n2;

			if (fscanf(fp, "%31s %31s %d", w1, w2, &n2) == 3) {
				feat = font_findfeat(f, "kern");
				r = (struct frule *)font_gpos(f, 2, 0, feat, -1, -1);
				x = dict_get(f->d78, w1);
				code = -1;
				if (x >= 0) {
					struct fglyph *g = f->glyph + x;
					if (g != NULL)
						code = g - f->glyph;
				}
				r->ent[0].code = code;
				x = dict_get(f->d78, w2);
				code = -1;
				if (x >= 0) {
					struct fglyph *g = f->glyph + x;
					if (g != NULL)
						code = g - f->glyph;
				}
				r->ent[1].code = code;
				r->ent[0].flags = 1;
				((short *)r->ent)[5] = n2;
				r->ent[1].flags = 1;
			}
		} else if (strcmp(buf, "ligatures") == 0) {
			for (;;) {
				if (fscanf(fp, "%s", lig[nlig]) != 1)
					break;
				if (strcmp(lig[nlig], "0") == 0)
					break;
				if (nlig < 512)
					nlig++;
			}
		} else if (strcmp(buf, "gsec") == 0) {
			char w1[32], w2[128];

			fgets(buf, 128, fp);
			n = sscanf(buf, "%d %15s %127s", &sec, w1, w2);
			if (n <= 1)
				continue;
			sel = strcmp(w1, "gpos") == 0;
			if (n > 2)
				font_readfeat(f, w2, &fi, &sc, &lg);
			continue;
		} else if (strcmp(buf, "gsub") == 0) {
			if (fscanf(fp, "%127s %d", buf, &b) == 2) {
				font_readfeat(f, buf, &fi, &sc, &lg);
				font_readgsub(f, fp, b, sec, fi, sc, lg);
			}
		} else if (strcmp(buf, "gpos") == 0) {
			if (fscanf(fp, "%127s %d", buf, &b) == 2) {
				font_readfeat(f, buf, &fi, &sc, &lg);
				font_readgpos(f, fp, b, sec, fi, sc, lg);
			}
		} else if (isdigit((unsigned char)buf[0])) {
			n = strtol(buf, NULL, 10);
			if (sel)
				font_readgpos(f, fp, n, sec, fi, sc, lg);
			else
				font_readgsub(f, fp, n, sec, fi, sc, lg);
		} else if (strcmp(buf, "ggrp") == 0) {
			int n2, cnt;
			char w2[128];

			if (fscanf(fp, "%d %d", &n2, &cnt) == 2) {
				for (i = 0; i < cnt; i++) {
					if (fscanf(fp, "%31s", w2) != 1)
						break;
					if (isdigit((unsigned char)w2[0])) {
						code = strtol(w2, NULL, 10);
						if (code < 0)
							continue;
					} else {
						x = dict_get(f->d78, w2);
						if (x < 0)
							continue;
						{
							struct fglyph *g = f->glyph + x;
							if (g == NULL)
								continue;
							code = g - f->glyph;
						}
					}
					if (code >= f->nglyph)
						continue;
					iset_put(f->iac8, n2, code);
					iset_put(f->iad0, code, n2);
				}
			}
		} else if (strcmp(buf, "spacewidth") == 0) {
			fscanf(fp, "%d", &f->spacewidth);
		} else if (strcmp(buf, "special") == 0) {
			f->special = 1;
		} else if (strcmp(buf, "name") == 0) {
			fscanf(fp, "%s", f->name);
		} else if (strcmp(buf, "fontname") == 0) {
			fscanf(fp, "%s", f->fontname);
		} else if (strcmp(buf, "charset") == 0) {
			while (font_readchar(f, fp, &c2, &c1) == 0) {
				for (;;) {
					int ch = getc(fp);

					if (ch == '\n' || ch == -1)
						break;
				}
			}
			break;
		}
		/* skip the rest of the line */
		for (;;) {
			int ch = getc(fp);

			if (ch == '\n' || ch == -1)
				break;
		}
	}

	if (nlig > 0) {
		char *s;
		char w1[32];
		int codes[256];

		for (i = 0; i < nlig; i++) {
			s = lig[i];
			k = 0;
			while (utf8read(&s, w1) > 0) {
				x = dict_get(f->d88, w1);
				if (x == -1)
					x = dict_get(f->d80, w1);
				code = -1;
				if (x >= 0) {
					struct fglyph *g = f->glyph + x;
					if (g != NULL)
						code = g - f->glyph;
				}
				codes[k++] = code;
			}
			feat = font_findfeat(f, "liga");
			r = (struct frule *)font_gsub(f, k + 1, 0, feat, -1, -1);
			for (j = 0; j < k; j++) {
				r->ent[j].code = codes[j];
				r->ent[j].flags = 1;
			}
			x = dict_get(f->d88, lig[i]);
			if (x == -1)
				x = dict_get(f->d80, lig[i]);
			code = -1;
			if (x >= 0) {
				struct fglyph *g = f->glyph + x;
				if (g != NULL)
					code = g - f->glyph;
			}
			r->ent[k].code = code;
			r->ent[k].flags = 2;
		}
	}

	fclose(fp);
	f->iab8 = iset_make();
	f->iac0 = iset_make();
	for (i = 0; i < f->ngsub; i++) {
		for (j = 0; j < f->gsub[i].n; j++) {
			if (!(f->gsub[i].ent[j].flags & 6)) {
				font_isetinsert(f, f->iab8, i, f->gsub[i].ent[j].code);
				break;
			}
		}
	}
	for (i = 0; i < f->ngpos; i++) {
		for (j = 0; j < f->gpos[i].n; j++) {
			if (!(f->gpos[i].ent[j].flags & 6)) {
				font_isetinsert(f, f->iac0, i, f->gpos[i].ent[j].code);
				break;
			}
		}
	}
	f->scrp = -1;
	f->lang = -1;
	return f;
}
