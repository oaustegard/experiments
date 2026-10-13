#include <string.h>
#include "mkfn_proto.h"
#include "uwid.h"
#include "otf_table.h"
#include "cff_char.h"
#include "cffdict_get.h"
#include "otf_gsub.h"
#include "otf_gpos.h"

#define U16(p) (((p)[0] << 8) | (p)[1])
#define U32(p) (((unsigned)(p)[0] << 24) | ((p)[1] << 16) | ((p)[2] << 8) | (p)[3])

int
otf_offsettable(const unsigned char *base, const unsigned char *tab)
{
	char fontname[128] = { 0 }, tmp[128];
	const unsigned char *head, *name, *cmap, *post, *glyf, *maxp, *loca, *cff;
	const unsigned char *kern, *p, *end, *sub;
	unsigned int tg;
	int n, i, j, k, len, off, s, e, fmt, idx, stroff;

	tg = *(const unsigned int *)tab;
	if (tg == 0x100 || tg == 0x4f54544f) {
		head = otf_table(base, tab, "head", tab[1]);
		upm = U16(head + 18);
		name = otf_table(base, tab, "name", tab[1]);
		if (U16(name + 2) != 0) {
			stroff = U16(name + 4);
			end = name + 6 + 12 * U16(name + 2);
			for (p = name + 6; p != end; p += 12) {
				if (U16(p) == 1 && (U16(p + 2) | U16(p + 4)) == 0 && U16(p + 6) == 6) {
					len = U16(p + 8);
					off = U16(p + 10);
					memcpy(tmp, name + stroff + off, len);
					tmp[len] = 0;
					snprintf(fontname, 0x80, "%s", tmp);
				}
			}
		}
		cmap = otf_table(base, tab, "cmap", tab[1]);
		if (U16(cmap + 2) != 0) {
			for (i = 0; i < U16(cmap + 2); i++) {
				const unsigned char *rec = cmap + 4 + 8 * i;
				if (U16(rec) == 3 && U16(rec + 2) == 1) {
					sub = cmap + U32(rec + 4);
					if (U16(sub) == 4) {
						int segx2 = U16(sub + 6);
						int seg = segx2 / 2;
						const unsigned char *ends = sub + 14;
						const unsigned char *starts = ends + segx2 + 2;
						const unsigned char *deltas = starts + segx2;
						const unsigned char *ranges = deltas + segx2;
						for (j = 0; j < seg; j++) {
							int c0 = U16(starts + 2 * j), c1 = U16(ends + 2 * j);
							int d = U16(deltas + 2 * j);
							int r = U16(ranges + 2 * j);
							if (r == 0) {
								for (k = c0; k <= c1; k++)
									glyph_code[(k + d) & 0xffff] = k;
							} else {
								for (k = c0; k <= c1; k++) {
									int g = U16(ranges + 2 * j + r + 2 * (k - c0));
									if (g)
										glyph_code[(g + d) & 0xffff] = k;
								}
							}
						}
					}
				}
			}
		}
		post = otf_table(base, tab, "post", tab[1]);
		if (*(const unsigned int *)post == 0x200) {
			n = U16(post + 0x20);
			glyph_n = n;
			p = post + 0x22 + 2 * n;
			k = 0;
			for (i = 0; i < n; i++) {
				idx = U16(post + 0x22 + 2 * i);
				if (idx <= 257) {
					strcpy(glyph_name[i], macset[idx]);
				} else {
					len = p[k];
					memcpy(glyph_name[i], p + k + 1, len);
					glyph_name[i][len] = 0;
					k += 1 + len;
				}
			}
		}
		glyf = otf_table(base, tab, "glyf", tab[1]);
		if (glyf) {
			maxp = otf_table(base, tab, "maxp", tab[1]);
			head = otf_table(base, tab, "head", tab[1]);
			loca = otf_table(base, tab, "loca", tab[1]);
			fmt = U16(head + 0x32);
			n = U16(maxp + 4);
			if (glyph_n == 0)
				glyph_n = n;
			for (i = 0; i < n; i++) {
				if (fmt) {
					s = U32(loca + 4 * i);
					e = U32(loca + 4 * i + 4);
				} else {
					s = 2 * U16(loca + 2 * i);
					e = 2 * U16(loca + 2 * i + 2);
				}
				if (s < e) {
					glyph_bbox[i][0] = (short)U16(glyf + s + 2);
					glyph_bbox[i][1] = (short)U16(glyf + s + 4);
					glyph_bbox[i][2] = (short)U16(glyf + s + 6);
					glyph_bbox[i][3] = (short)U16(glyf + s + 8);
				}
			}
		}
		cff = otf_table(base, tab, "CFF ", tab[1]);
		if (cff && cff[0] == 1) {
			const unsigned char *idx0 = cff + cff[2];
			const unsigned char *top, *strs, *cs, *dict;
			int nn = U16(idx0), osz = idx0[2];
			int last = 0, ntop, otop, toff0, toff1, dlen, cs_off, cf;
			for (j = 0; j < osz; j++)
				last = (last << 8) + idx0[3 + nn * osz + j];
			top = idx0 + 2 + (nn + 1) * osz + last;
			ntop = U16(top);
			otop = top[2];
			toff0 = 0;
			for (j = 0; j < otop; j++)
				toff0 = (toff0 << 8) + top[3 + j];
			toff1 = 0;
			for (j = 0; j < otop; j++)
				toff1 = (toff1 << 8) + top[3 + otop + j];
			dlen = toff1 - toff0;
			dict = top + 2 + (ntop + 1) * otop + toff0;
			cs_off = cffdict_get(dict, dlen, 17);
			cs = cff + cs_off;
			strs = top + 2 + (ntop + 1) * otop + toff1;
			cf = cffdict_get(dict, dlen, 15);
			cs = cff + cf;
			if (cs[0] == 0) {
				for (i = 0; i < glyph_n; i++)
					cff_char(strs, U16(cs + 1 + 2 * i), glyph_name[i + 1]);
			} else {
				for (i = 1, j = 1; i < glyph_n; ) {
					int first = U16(cs + j);
					int nl = cs[0] == 1 ? cs[j + 2] : U16(cs + j + 2);
					int m;
					for (m = 0; m <= nl && i < glyph_n; m++, i++)
						cff_char(strs, first + m, glyph_name[i]);
					j += cs[0] == 1 ? 3 : 4;
				}
			}
		}
		for (i = 0; i < glyph_n; i++) {
			if (glyph_name[i][0] == 0) {
				if (glyph_code[i])
					sprintf(glyph_name[i], "uni%04X", glyph_code[i]);
				else
					sprintf(glyph_name[i], "gl%05X", i);
			}
		}
		{
			const unsigned char *hmtx = otf_table(base, tab, "hmtx", tab[1]);
			const unsigned char *hhea = otf_table(base, tab, "hhea", tab[1]);
			int nh = U16(hhea + 0x22);
			for (i = 0; i < nh; i++)
				glyph_wid[i] = U16(hmtx + 4 * i);
			for (i = nh; i < glyph_n; i++)
				glyph_wid[i] = glyph_wid[nh - 1];
		}
		if (!mkfn_font(fontname))
			return 0;
		for (i = 0; i < glyph_n; i++) {
			int c = glyph_code[i];
			if (c == 0xffff)
				c = 0;
			mkfn_char(glyph_name[i], -1, c, uwid(glyph_wid[i]), uwid(glyph_bbox[i][0]),
				uwid(glyph_bbox[i][1]), uwid(glyph_bbox[i][2]), uwid(glyph_bbox[i][3]));
		}
		mkfn_header(fontname);
		kern = otf_table(base, tab, "kern", tab[1]);
		if (kern && U16(kern + 2) != 0) {
			int nt = U16(kern + 2);
			int off2 = 4;
			for (i = 0; i < nt; i++) {
				const unsigned char *st = kern + off2;
				int cov = (st[4] << 8) | st[5];
				off2 += U16(st + 2);
				if ((cov & 0xff01) == 0x0001 && U16(st + 6) != 0) {
					int np = U16(st + 6);
					const unsigned char *pp;
					for (pp = st + 14; pp != st + 14 + 6 * np; pp += 6) {
						int v = uwid((short)U16(pp + 4));
						mkfn_kern(glyph_name[U16(pp)], glyph_name[U16(pp + 2)], v);
					}
				}
			}
		}
		{
			const unsigned char *gt = otf_table(base, tab, "GSUB", tab[1]);
			if (gt)
				otf_gsub(gt, tab[1]);
		}
		{
			const unsigned char *pt = otf_table(base, tab, "GPOS", tab[1]);
			if (pt)
				otf_gpos(pt, tab[1]);
		}
		return 0;
	}
	return 1;
}
