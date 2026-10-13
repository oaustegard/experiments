#include <ctype.h>
#include <string.h>
#include "sbuf.h"
#include "tab.h"
#include "achar.h"
#include "str_pair.h"
#include "fn_sbuf_printf.h"
#include "fn_tab_get.h"
#include "mkfn_pos.h"
#include "mkfn_desc.h"
#include "mkfn_asc.h"
#include "mkfn_bbox.h"
#include "mkfn_swid.h"
#include "mkfn_ligs.h"
#include "mkfn_ligs2.h"

void mkfn_char(char *s, int code, int uni, int wid, int ht, int dp, int lig, int bbox)
{
	char nm[64], desc[64] = {0}, key[64];
	char *u = nm, *e = nm, *p, *q, *r, *b, *w, *h;
	char **a;
	int ucs[8];
	int v, i, n, k, fl, ini, prev, next, c;
	size_t len;

	if (uni) {
		utf8put(&u, uni);
		goto agl;
	}
	if (s == 0 || *s == '.') {
		strcpy(nm, "---");
		goto agl;
	}
	q = s;
	if (*q) {
		for (;;) {
			b = e;
			if (*q == '_')
				q++;
			h = key;
			if (*q && *q != '_' && *q != '.') {
				do
					*h++ = *q++;
				while (*q && *q != '_' && *q != '.');
			}
			*h = 0;
			p = b;
			r = tab_get(tab_agl, key);
			if (r) {
				while (*r) {
					while (*r == ' ')
						r++;
					v = 0;
					for (n = 0; r[n] && strchr("0123456789ABCDEF", r[n]); ) {
						v = v * 16 + (int)(strchr("0123456789ABCDEF", r[n]) - "0123456789ABCDEF");
						if (++n == 6)
							break;
					}
					if (n <= 3)
						v = -1;
					utf8put(&p, v);
					r = strchr(r, ' ');
					if (r == 0)
						break;
				}
				*p = 0;
				e = b + strlen(b);
			} else if (key[0] == 'u') {
				if (key[1] == 'n' && key[2] == 'i') {
					v = 0;
					for (n = 0; n < 4 && key[3 + n] && strchr("0123456789ABCDEF", key[3 + n]); n++)
						v = v * 16 + (int)(strchr("0123456789ABCDEF", key[3 + n]) - "0123456789ABCDEF");
					if (v > 0) {
						for (n = 0; strlen(key + 3) > 3; ) {
							v = 0;
							for (i = 0; i < 4; i++) {
								c = key[3 + n + i];
								if (c == 0 || strchr("0123456789ABCDEF", c) == 0) {
									v = -1;
									break;
								}
								v = v * 16 + (int)(strchr("0123456789ABCDEF", c) - "0123456789ABCDEF");
							}
							utf8put(&e, v);
							n += 4;
							if (n > 0)
								break;
						}
						goto next;
					}
				} else {
					v = 0;
					for (n = 0; n < 4 && key[1 + n] && strchr("0123456789ABCDEF", key[1 + n]); n++)
						v = v * 16 + (int)(strchr("0123456789ABCDEF", key[1 + n]) - "0123456789ABCDEF");
					if (v > 0) {
						v = 0;
						for (n = 0; n < 6 && key[1 + n] && strchr("0123456789ABCDEF", key[1 + n]); n++)
							v = v * 16 + (int)(strchr("0123456789ABCDEF", key[1 + n]) - "0123456789ABCDEF");
						if (n <= 3)
							v = -1;
						utf8put(&e, v);
						goto next;
					}
				}
			}
			{
				/* achars search */
				for (i = 0, len = 5; i < 79; i++) {
					if (i)
						len = strlen(achars[i].name);
					if (strncmp(achars[i].name, key, len) == 0) {
						const char *sfx = key + len;
						if (*sfx == 0)
							c = achars[i].code;
						else if (strcmp(sfx, "isolated") == 0)
							c = achars[i].isol ? achars[i].isol : achars[i].code;
						else if (strcmp(sfx, "initial") == 0)
							c = achars[i].init ? achars[i].init : achars[i].code;
						else if (strcmp(sfx, "medial") == 0)
							c = achars[i].medi ? achars[i].medi : achars[i].code;
						else if (strcmp(sfx, "final") == 0)
							c = achars[i].fina ? achars[i].fina : achars[i].code;
						else
							continue;
						if (c == 0)
							break;
						utf8put(&e, c);
						goto next;
					}
				}
			}
			strcpy(nm, "---");
			break;
		next:
			if (*q && *q != '.')
				continue;
			break;
		}
	}
	/* suffix forms */
	w = nm;
	if (strcmp(".medi", q) == 0) {
		fl = 1;
		ini = 1;
	} else {
		fl = strcmp(".fina", q) == 0;
		ini = strcmp(".init", q) == 0;
	}
	for (i = 0, w = nm; *w && i < 8; i++) {
		c = (unsigned char)*w++;
		if ((c & 0xc0) != 0xc0) {
			ucs[i] = c;
		} else {
			int cont = (c & 0x20) ? ((c & 0x10) ? 3 : 2) : 1;
			int val = c & (0x3f >> cont);
			while (cont-- > 0 && *w)
				val = (val << 6) | (*w++ & 0x3f);
			ucs[i] = val;
		}
	}
	n = i;
	if (n == 0) {
		if (*q == 0)
			goto agl;
		if (strcmp(q, ".medi") == 0 || strcmp(q, ".fina") == 0 || strcmp(q, ".init") == 0 || strcmp(q, ".isol") == 0)
			goto agl;
		strcpy(nm, "---");
		goto agl;
	}
	for (k = 0; k < n; k++) {
		prev = (k > 0) || fl;
		next = (k < n - 1) || ini;
		for (i = 0; i < 79; i++)
			if (achars[i].code == ucs[k])
				break;
		if (i < 79) {
			if (!prev && !next)
				continue;
			if (!prev)
				c = achars[i].init;
			else if (!next)
				c = achars[i].fina;
			else
				c = achars[i].medi;
			if (c == 0)
				c = achars[i].code;
			ucs[k] = c;
		}
	}
	w = nm;
	for (k = 0; k < n; k++)
		utf8put(&w, ucs[k]);
	e = w;
agl:
	for (i = 0; i < 4; i++) {
		if (strcmp(agl_exceptions[i].from, nm) == 0)
			strcpy(nm, agl_exceptions[i].to);
	}
	if (mkfn_pos) {
		if (code > 0xff)
			sprintf(desc, "%d", code);
		else
			sprintf(desc, "%d", code);
	}
	if (isspace((unsigned char)nm[0]) || strchr(nm, ' '))
		strcpy(nm, "---");
	if (strcmp(nm, "---") != 0) {
		for (i = 0; i < 6; i++)
			if (strcmp(ligs_utf8[i].from, nm) == 0)
				sprintf(mkfn_ligs, "%s ", ligs_utf8[i].to);
	}
	k = 3;
	if (strchr(s, '.') && mkfn_desc && mkfn_asc) {
		k = dp <= mkfn_desc;
		if (bbox >= mkfn_asc)
			k |= 2;
	}
	if (!mkfn_swid && strcmp(nm, " ") == 0)
		mkfn_swid = wid;
	sbuf_printf(sbuf_char, "char %s\t%d", nm, wid);
	if (mkfn_bbox && (ht | dp | lig | bbox))
		sbuf_printf(sbuf_char, ",%d,%d,%d,%d", ht, dp, lig, bbox);
	sbuf_printf(sbuf_char, "\t%d\t%s\t%s\n", k, s, desc);
	a = tab_get(tab_alts, nm);
	if (a)
		for (; *a; a++)
			sbuf_printf(sbuf_char, "char %s\t\"\n", *a);
}
