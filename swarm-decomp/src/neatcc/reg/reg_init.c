#include "region.h"
#include "ic_ent.h"
#include "opt.h"
#include "argregs.h"
#include "rgn_add.h"
#include "reg_region.h"

void reg_init(struct ic_ent *tab, long n)
{
	struct ic_ent *p;
	long i, j, k, q, s, a, b, x, c, lo, hi, key, w, mask, count;
	int nl = loc_n, grew = 0, *pref, *order, regs[4], nregs = 0, r10, start, t;
	char *seen, ok = 1;

	if (n > 0) {
		for (i = 0; i < n; i++) {
			p = tab + i;
			if (!(p->op & 0x200000))
				continue;
			k = p->op & 0xffffff;
			if (k == 0x200400 || k == 0x210000)
				x = p->a;
			else if (k == 0x220000)
				x = p->b;
			else
				continue;
			if (nl <= x + 1) {
				nl = x + 1;
				grew = 1;
			}
		}
		if (grew)
			loc_n = nl;
	}
	loc_ptr = calloc(nl, 4);
	pref = calloc(nl, 4);
	if (nl > 0)
		for (i = 0; i < loc_n; i++)
			loc_ptr[i] = !opt(1);
	if (n > 0) {
		for (p = tab; p != tab + n; p++) {
			k = p->op & 0xffffff;
			if (k == 0x210000 || k == 0x200400) {
				a = p->a;
				b = p->b;
				if (k == 0x200400) {
					loc_ptr[a]++;
					continue;
				}
			} else if (k == 0x220000) {
				a = p->b;
				b = p->c;
			} else {
				continue;
			}
			t = (p->op >> 24) & 0xf;
			if (pref[a] == 0)
				pref[a] = t;
			if (b != 0 || ((p->op >> 24) & 0xe) == 0 || pref[a] != t)
				loc_ptr[a]++;
		}
	}
	free(pref);
	ok = 1;
	if (n > 0) {
		for (p = tab; p != tab + n; p++)
			if (p->op & 0x200)
				ok = 0;
	}
	dst_head = malloc(n * 8);
	dst_next = malloc(n * 8);
	memset(dst_head, 0xff, n * 8);
	memset(dst_next, 0xff, n * 8);
	if (n > 0) {
		for (p = tab, i = 0; i < n; i++, p++) {
			if (p->op & 0x7000) {
				c = p->c;
				j = dst_head[c];
				dst_head[c] = i;
				dst_next[i] = j;
			}
		}
	}
	if (loc_n > 0) {
		for (i = 0; i < loc_n; i++) {
			if (loc_ptr[i])
				continue;
			if (opt(2)) {
				seen = calloc(n, 1);
				for (k = 0; k < n; k++) {
					p = tab + k;
					if ((p->op & 0xffffff) != 0x210000 || p->a != i || seen[k])
						continue;
					lo = k;
					hi = k + 1;
					count = reg_region(tab, i, k, &lo, &hi, seen, n);
					rgn_add(i, lo, hi, count);
				}
				for (k = 0; k < n; k++) {
					p = tab + k;
					if ((p->op & 0xffffff) != 0x220000 || p->b != i || seen[k])
						continue;
					rgn_add(i, k, k + 1, 1);
				}
				free(seen);
			}
			if (loc_ptr[i])
				continue;
			if (opt(2))
				continue;
			count = 0;
			for (p = tab; p != tab + n; p++) {
				if ((p->op & 0xffffff) == 0x210000) {
					if (p->a == i)
						count++;
				} else if ((p->op & 0xffffff) == 0x220000 && p->b == i) {
					count++;
				}
			}
			rgn_add(i, 0, n, count);
		}
	}
	if (ok) {
		start = 1;
		mask = 0xffcf;
	} else {
		start = 3;
		mask = 0xf008;
	}
	for (r10 = start; r10 <= 13 && nregs < 4; r10++)
		if (mask & (1 << r10))
			regs[nregs++] = r10;
	order = malloc(rgn_n * 4);
	if (rgn_n > 0) {
		for (j = 0; j < rgn_n; j++) {
			w = rgn[j].weight;
			for (k = j - 1; k >= 0 && w > rgn[order[k]].weight; k--)
				order[k + 1] = order[k];
			order[k + 1] = j;
		}
		for (t = 0; t < rgn_n; t++) {
			q = order[t];
			key = rgn[q].key;
			if (key < 0 || loc_ptr[key])
				continue;
			lo = rgn[q].lo;
			hi = rgn[q].hi;
			if (lo == 0 && key <= 5 && ok) {
				r10 = argregs[key];
				for (k = 0; k < rgn_n; k++)
					if (rgn[k].reg == r10 && hi > rgn[k].lo && rgn[k].hi > 0)
						break;
				if (k == rgn_n) {
					rgn[q].reg = r10;
					continue;
				}
			}
			for (s = 0; s < nregs; s++) {
				for (k = 0; k < rgn_n; k++)
					if (rgn[k].reg == regs[s] && hi > rgn[k].lo && lo < rgn[k].hi)
						break;
				if (k == rgn_n) {
					rgn[q].reg = regs[s];
					break;
				}
			}
		}
	}
	free(order);
}
