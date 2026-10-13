#include <stdlib.h>
#include <string.h>
#include "otf_feat.h"
#include "otf_gtab.h"
#include "coverage.h"
#include "ggrp_find.h"
#include "ggrp_make.h"
#include "intcmp.h"
#include "lookuptag.h"
#include "mkfn_byname.h"
#include "mkfn_dry.h"
#include "mkfn_proto.h"
#include "sec.h"
#include "uwid.h"

#define U16(p) ({ unsigned short w_ = *(const unsigned short *)(p); (int)(unsigned short)((w_ << 8) | (w_ >> 8)); })
#define U32(p) ({ unsigned w_ = *(const unsigned *)(p); (w_ << 24) | ((w_ << 8) & 0xff0000u) | ((w_ >> 8) & 0xff00u) | (w_ >> 24); })

/* glyph name for printing: numeric when names are not used */
#define GNAME(g) \
	(!mkfn_byname && strlen(glyph_name[g]) > 3 \
	    ? (sprintf(str, "%d", (g)), str) : glyph_name[g])

/* number of bytes of a value record for the value format mask */
#define VSIZE(m, sz) \
	do { \
		int b_; \
		(sz) = 0; \
		for (b_ = 0; b_ < 8; b_++) \
			if ((m) & (1 << b_)) \
				(sz) += 2; \
	} while (0)

static void
otf_gpos(const unsigned char *t, int unused)
{
	struct otf_feat ents[4096];
	static int cls1[65536], gl1[65536], cls2[65536], gl2[65536];
	static int grp1[65536], grp2[65536];
	static char str[16];
	struct otf_feat *f;
	const unsigned char *ll, *lk, *sub, *p;
	int *g, *bg, *lst;
	const unsigned char *ma;
	int n1, n2, nm;
	int n, s, type, ty, nsub, fmt, vf1, vf2, sz1, sz2, cnt, vc, off, mask;
	int i, j, k, c1, c2, ne, nx, ncount;
	char *tag;

	ll = t + U16(t + 8);
	n = otf_gtab(t, ents, t[1]);
	if (mkfn_dry || n <= 0)
		return;
	for (f = ents, s = 10; f != ents + n; f++, s += 10) {
		lk = ll + U16(ll + 2 + 2 * f->idx);
		type = U16(lk);
		nsub = U16(lk + 4);
		tag = lookuptag(f->name);
		sec = s;
		printf("gsec %d gpos %s\n", sec, tag);
		for (p = lk + 6; p != lk + 6 + 2 * nsub; p += 2) {
			sub = lk + U16(p);
			ty = type;
			if (ty == 9) {
				ty = U16(sub + 2);
				sub += U32(sub + 4);
			}
			switch (ty) {
			case 1:
				{
					int cnt;
					fmt = U16(sub);
					mask = U16(sub + 4);
					VSIZE(mask, sz1);
					g = coverage(sub + U16(sub + 2), &cnt);
					if (fmt == 1) {
						for (i = 0; i < cnt; i++) {
							if (!valuerecord_small(mask, sub + 6)) {
								printf("1 %s", GNAME(g[i]));
								valuerecord_print(mask, sub + 6);
								putchar('\n');
							}
						}
					} else if (fmt == 2) {
						vc = U16(sub + 6);
						if (vc != 0) {
							for (i = 0, off = 0; i < vc; i++, off += sz1) {
								if (!valuerecord_small(mask, sub + 6)) {
									printf("1 %s", GNAME(g[i]));
									valuerecord_print(mask, sub + 8 + off);
									putchar('\n');
								}
							}
						}
					}
					free(g);
				}
				break;
			case 2:
				fmt = U16(sub);
				vf1 = U16(sub + 4);
				vf2 = U16(sub + 6);
				VSIZE(vf1, sz1);
				VSIZE(vf2, sz2);
				if (fmt == 1) {
					g = coverage(sub + U16(sub + 2), NULL);
					ncount = U16(sub + 8);
					for (i = 0; i < ncount; i++) {
						p = sub + U16(sub + 10 + 2 * i);
						vc = U16(p);
						for (k = 0, off = 0; k < vc; k++, off += sz1 + sz2 + 2) {
							if (!valuerecord_small(vf1, p + 4 + off) ||
							    !valuerecord_small(vf2, p + 4 + off + sz1)) {
								printf("2 %s", GNAME(g[i]));
								valuerecord_print(vf1, p + 4 + off);
								printf(" %s", GNAME(U16(p + 2 + off)));
								valuerecord_print(vf2, p + 4 + off + sz1);
								putchar('\n');
							}
						}
					}
					free(g);
				} else if (fmt == 2) {
					n1 = classdef(sub + U16(sub + 8), gl1, cls1);
					n2 = classdef(sub + U16(sub + 10), gl2, cls2);
					c1 = U16(sub + 12);
					c2 = U16(sub + 14);
					for (i = 0; i < c1; i++)
						grp1[i] = ggrp_class(gl1, cls1, n1, i);
					for (j = 0; j < c2; j++)
						grp2[j] = ggrp_class(gl2, cls2, n2, j);
					for (i = 0; i < c1; i++) {
						for (j = 0; j < c2; j++) {
							const unsigned char *v1 = sub + 16 + (i * c2 + j) * (sz1 + sz2);
							const unsigned char *v2 = v1 + sz1;
							if (!valuerecord_small(vf1, v1) ||
							    !valuerecord_small(vf2, v2)) {
								printf("2 @%d", grp1[i]);
								valuerecord_print(vf1, v1);
								printf(" @%d", grp2[j]);
								valuerecord_print(vf2, v2);
								putchar('\n');
							}
						}
					}
				}
				break;
			case 3:
				if (U16(sub) == 1) {
					int *en, *ex;
					int nen, nex, nce, ncx, x, y, w;
					g = coverage(sub + U16(sub + 2), NULL);
					ne = U16(sub + 4);
					ex = malloc(4 * ne);
					en = malloc(4 * ne);
					nen = 0;
					for (i = 0; i < ne; i++)
						if (U16(sub + 6 + 4 * i) != 0)
							en[nen++] = g[i];
					nex = 0;
					for (i = 0; i < ne; i++)
						if (U16(sub + 8 + 4 * i) != 0)
							ex[nex++] = g[i];
					qsort(ex, nex, sizeof *ex, intcmp);
					ncx = ggrp_find(ex, nex);
					if (ncx < 0)
						ncx = ggrp_make(ex, nex);
					qsort(en, nen, sizeof *en, intcmp);
					nce = ggrp_find(en, nen);
					if (nce < 0)
						nce = ggrp_make(en, nen);
					free(ex);
					free(en);
					printf("gsec %d gpos %s\n", sec, tag);
					if (ne > 0) {
						for (i = 0; i < ne; i++) {
							const unsigned char *an;
							if (U16(sub + 6 + 4 * i) == 0)
								continue;
							an = sub + U16(sub + 6 + 4 * i);
							x = uwid((short)U16(an + 2));
							y = uwid((short)U16(an + 4));
							y = -y;
							if (strcmp("arab", strchr(tag, ':') + 1) == 0 ||
							    strcmp("hebr", strchr(tag, ':') + 1) == 0) {
								w = uwid(glyph_wid[g[i]]);
								x = w - x;
							} else
								x = -x;
							printf("2 @%d %s:%+d%+d%+d%+d\n", nce, GNAME(g[i]), 0, 0, x, y);
						}
					}
					printf("gsec %d gpos %s\n", sec + 1, tag);
					for (i = 0; i < ne; i++) {
						const unsigned char *an;
						if (U16(sub + 8 + 4 * i) == 0)
							continue;
						an = sub + U16(sub + 8 + 4 * i);
						x = uwid((short)U16(an + 2));
						w = uwid(glyph_wid[g[i]]);
						y = uwid((short)U16(an + 4));
						if (strcmp("arab", strchr(tag, ':') + 1) != 0 &&
						    strcmp("hebr", strchr(tag, ':') + 1) != 0)
							x = x - w;
						printf("2 %s @%d:%+d%+d%+d%+d\n", GNAME(g[i]), ncx, 0, 0, x, y);
					}
					free(g);
				}
				break;
			case 4:
				if (U16(sub) == 1) {
					int gi[1024];
					int b, x, y, w;
					const unsigned char *an, *bas;
					g = coverage(sub + U16(sub + 2), &n1);
					bg = coverage(sub + U16(sub + 4), &n2);
					c1 = U16(sub + 6);
					ma = sub + U16(sub + 8);
					bas = sub + U16(sub + 10);
					qsort(bg, n2, sizeof *bg, intcmp);
					ncount = ggrp_find(bg, n2);
					if (ncount < 0)
						ncount = ggrp_make(bg, n2);
					for (k = 0; k < c1; k++) {
						lst = malloc(4 * n1);
						nm = 0;
						for (i = 0; i < n1; i++)
							if (U16(ma + 2 + 4 * i) == k)
								lst[nm++] = g[i];
						qsort(lst, nm, sizeof *lst, intcmp);
						gi[k] = ggrp_find(lst, nm);
						if (gi[k] < 0)
							gi[k] = ggrp_make(lst, nm);
						free(lst);
					}
					printf("gsec %d gpos %s\n", sec, tag);
					for (i = 0; i < n1; i++) {
						an = ma + U16(ma + 4 + 4 * i);
						x = uwid((short)U16(an + 2));
						y = uwid((short)U16(an + 4));
						if (strcmp("arab", strchr(tag, ':') + 1) == 0 ||
						    strcmp("hebr", strchr(tag, ':') + 1) == 0) {
							w = uwid(glyph_wid[g[i]]);
							x = w - x;
						} else {
							x = -x;
							y = -y;
						}
						printf("2 @%d %s:%+d%+d%+d%+d\n", n2, GNAME(g[i]), x, y, 0, 0);
					}
					printf("gsec %d gpos %s\n", sec + 1, tag);
					for (b = 0; b < n2; b++) {
						for (k = 0; k < c1; k++) {
							an = bas + U16(bas + 2 + 2 * (b * c1 + k));
							x = uwid((short)U16(an + 2));
							w = uwid(glyph_wid[bg[b]]);
							y = uwid((short)U16(an + 4));
							if (strcmp("arab", strchr(tag, ':') + 1) == 0 ||
							    strcmp("hebr", strchr(tag, ':') + 1) == 0)
								y = -y;
							else
								x = x - w;
							printf("2 %s @%d:%+d%+d%+d%+d\n", GNAME(bg[b]), gi[k], x, y, 0, 0);
						}
					}
					free(g);
					free(bg);
				}
				break;
			case 5:
				if (U16(sub) == 1) {
					int gi[1024];
					int b, x, y, w, comps;
					const unsigned char *an, *lig, *la;
					g = coverage(sub + U16(sub + 2), &n1);
					bg = coverage(sub + U16(sub + 4), &n2);
					qsort(bg, n2, sizeof *bg, intcmp);
					ncount = ggrp_find(bg, n2);
					if (ncount < 0)
						ncount = ggrp_make(bg, n2);
					c1 = U16(sub + 6);
					ma = sub + U16(sub + 8);
					lig = sub + U16(sub + 10);
					for (k = 0; k < c1; k++) {
						lst = malloc(4 * n1);
						nm = 0;
						for (i = 0; i < n1; i++)
							if (U16(ma + 2 + 4 * i) == k)
								lst[nm++] = g[i];
						qsort(lst, nm, sizeof *lst, intcmp);
						gi[k] = ggrp_find(lst, nm);
						if (gi[k] < 0)
							gi[k] = ggrp_make(lst, nm);
						free(lst);
					}
					printf("gsec %d gpos %s\n", sec, tag);
					for (i = 0; i < n1; i++) {
						an = ma + U16(ma + 4 + 4 * i);
						x = uwid((short)U16(an + 2));
						y = uwid((short)U16(an + 4));
						if (strcmp("arab", strchr(tag, ':') + 1) == 0 ||
						    strcmp("hebr", strchr(tag, ':') + 1) == 0) {
							w = uwid(glyph_wid[g[i]]);
							x = w - x;
						} else {
							x = -x;
							y = -y;
						}
						printf("2 @%d %s:%+d%+d%+d%+d\n", n2, GNAME(g[i]), x, y, 0, 0);
					}
					printf("gsec %d gpos %s\n", sec + 1, tag);
					for (b = 0; b < n2; b++) {
						la = lig + U16(lig + 2 + 2 * b);
						comps = U16(la);
						if (comps == 0 || c1 == 0 ||
						    U16(la + 2 + 2 * (comps - 1) * c1) == 0)
							continue;
						for (k = 0; k < c1; k++) {
							an = la + U16(la + 2 + 2 * ((comps - 1) * c1 + k));
							x = uwid((short)U16(an + 2));
							w = uwid(glyph_wid[bg[b]]);
							y = uwid((short)U16(an + 4));
							if (strcmp("arab", strchr(tag, ':') + 1) == 0 ||
							    strcmp("hebr", strchr(tag, ':') + 1) == 0)
								y = -y;
							else
								x = x - w;
							printf("2 %s @%d:%+d%+d%+d%+d\n", GNAME(bg[b]), gi[k], x, y, 0, 0);
						}
					}
					free(g);
					free(bg);
				}
				break;
			default:
				if (mkfn_warn) {
					fprintf(stderr, "neatmkfn: unsupported %s lookup %d", "GPOS", ty);
					fputc('\n', stderr);
				}
				break;
			}
		}
	}
}
