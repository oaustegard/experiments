#include <stdlib.h>
#include "otf_feat.h"
#include "otf_gtab.h"
#include "otf_chain.h"
#include "coverage.h"
#include "ggrp_find.h"
#include "ggrp_make.h"
#include "intcmp.h"
#include "lookuptag.h"
#include "otf_gsubtype1.h"
#include "otf_gsubtype3.h"
#include "otf_gsubtype4.h"

#define U16(p) ({ unsigned short w_ = *(const unsigned short *)(p); (int)(unsigned short)((w_ << 8) | (w_ >> 8)); })
#define U32(p) ({ unsigned w_ = *(const unsigned *)(p); (w_ << 24) | ((w_ << 8) & 0xff0000u) | ((w_ >> 8) & 0xff00u) | (w_ >> 24); })

/* one coverage table of a chaining subtable: group index into dst */
#define GRP_OF(dst, off) \
	do { \
		int *g_, m_; \
		g_ = coverage(sub + (off), &m_); \
		qsort(g_, m_, sizeof *g_, intcmp); \
		(dst) = ggrp_find(g_, m_); \
		if ((dst) < 0) \
			(dst) = ggrp_make(g_, m_); \
		free(g_); \
	} while (0)

static void
otf_gsub(const unsigned char *t, int unused)
{
	struct otf_feat ents[4096];
	struct otf_feat *f;
	const unsigned char *ll, *lk, *lk2, *ll2, *sub, *sub2, *p;
	int n, k, s, type, ty, nsub, fmt, nb, ni, nl, pos;
	int type2, ty2, nsub2, j;

	ll = t + U16(t + 8);
	n = otf_gtab(t, ents, t[0]);
	if (mkfn_dry || n <= 0)
		return;
	for (f = ents, s = 10; f != ents + n; f++, s += 10) {
		lk = ll + U16(ll + 2 + 2 * f->idx);
		type = U16(lk);
		nsub = U16(lk + 4);
		printf("gsec %d gsub %s\n", sec = s, lookuptag(f->name));
		for (p = lk + 6; p != lk + 6 + 2 * nsub; p += 2) {
			sub = lk + U16(p);
			ty = type;
			if (ty == 7) {
				ty = U16(sub + 2);
				sub += U32(sub + 4);
			}
			switch (ty) {
			case 1:
				otf_gsubtype1(sub, NULL, type);
				break;
			case 3:
				otf_gsubtype3(sub, NULL);
				break;
			case 4:
				otf_gsubtype4(sub, NULL);
				break;
			case 6:
				{
					struct otf_chain c = {0};

					ll2 = t + U16(t + 8);
					fmt = U16(sub);
					if (fmt == 3) {
						nb = U16(sub + 2);
						c.nb = nb;
						for (j = 0; j < nb; j++)
							GRP_OF(c.bt[j], U16(sub + 4 + 2 * j));
						ni = U16(sub + 4 + 2 * nb);
						c.ni = ni;
						for (j = 0; j < ni; j++)
							GRP_OF(c.in[j], U16(sub + 6 + 2 * nb + 2 * j));
						pos = 6 + 2 * nb + 2 * ni;
						nl = U16(sub + pos);
						c.nl = nl;
						for (j = 0; j < nl; j++)
							GRP_OF(c.la[j], U16(sub + pos + 2 + 2 * j));
						pos = pos + 2 + 2 * nl;
						if (U16(sub + pos) != 0) {
							c.seq = U16(sub + pos + 2);
							lk2 = ll2 + U16(ll2 + 2 + 2 * U16(sub + pos + 4));
							type2 = U16(lk2);
							nsub2 = U16(lk2 + 4);
							for (j = 0; j < nsub2; j++) {
								sub2 = lk2 + U16(lk2 + 6 + 2 * j);
								ty2 = type2;
								if (ty2 == 7) {
									ty2 = U16(sub2 + 2);
									sub2 += U32(sub2 + 4);
								}
								switch (ty2) {
								case 1:
									otf_gsubtype1(sub2, &c, type2);
									break;
								case 3:
									otf_gsubtype3(sub2, &c);
									break;
								case 4:
									otf_gsubtype4(sub2, &c);
									break;
								}
							}
						}
					} else if (mkfn_warn) {
						fprintf(stderr, "neatmkfn: unsupported %s lookup %d", "GSUB", 6);
						if (fmt)
							fprintf(stderr, " format %d", fmt);
						fputc('\n', stderr);
					}
				}
				break;
			default:
				if (mkfn_warn) {
					fprintf(stderr, "neatmkfn: unsupported %s lookup %d", "GSUB", ty);
					fputc('\n', stderr);
				}
				break;
			}
		}
	}
}
