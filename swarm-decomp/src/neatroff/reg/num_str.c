#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "env.h"
#include "fn_map_name.h"
#include "fn_nreg.h"
#include "fn_dev_font.h"
#include "fn_font_getbd.h"
#include "fn_in_filename.h"
#include "fn_in_lnum.h"
#include "fn_in_nargs.h"
#include "fn_f_nexttrap.h"
#include "fn_f_hpos.h"
#include "fn_f_divreg.h"

char *num_str(int r)
{
	static char numbuf[128];
	static char tabs[512];
	static char dst[1024];
	char buf[128];
	char *s, *p, *e;
	int v, d, c, i, n, ei;
	unsigned char *q;

	s = map_name(r);
	if (!nregs_fmt[r])
		nregs_fmt[r] = '0';
	numbuf[0] = 0;
	if (s[0] == '.') {
		if (s[1] && !s[2]) {
			if (s[1] == '$') {
				sprintf(numbuf, "%d", in_nargs() - 1);
				return numbuf;
			}
			switch (s[1]) {
			case 'D':
				p = in_filename();
				e = strrchr(p, '/');
				if (!e)
					p = ".";
				else if (e == p)
					p = "/";
				else {
					i = e - p;
					memcpy(dst, p, i);
					dst[i] = 0;
					p = dst;
				}
				snprintf(numbuf, 0x80, "%s", p);
				return numbuf;
			case 'F':
				snprintf(numbuf, 0x80, "%s", in_filename());
				return numbuf;
			case 'b':
				sprintf(numbuf, "%d", font_getbd(dev_font(*nreg('f'))));
				return numbuf;
			case 'c':
				sprintf(numbuf, "%d", in_lnum());
				return numbuf;
			case 'k':
				sprintf(numbuf, "%d", f_hpos());
				return numbuf;
			case 'm':
				v = *nreg('m');
				sprintf(numbuf, "#%02x%02x%02x", (v >> 16) & 0xff, (v >> 8) & 0xff, v & 0xff);
				return numbuf;
			case 't':
				sprintf(numbuf, "%d", f_nexttrap());
				return numbuf;
			case 'z':
				if (f_divreg() < 0)
					return numbuf;
				snprintf(numbuf, 0x80, "%s", map_name(f_divreg()));
				return numbuf;
			}
		} else if (s[1] && s[2]) {
			if (!strcmp(".neat", s))
				return "1";
			if (s[1] == 'e' && s[2] == 'v' && !s[3]) 
				return map_name(env_id);
		}
		if (!strcmp(".neat", s))
			return "1";
		if (!strcmp(".tabs", s)) {
			p = tabs;
			for (i = 0; (c = env->pad180[i]) != 0; ) {
				p += sprintf(p, "%du%c ", env->tab[i], c);
				i++;
				if (i == 32)
					break;
			}
			return tabs;
		}
	} else if (s[0] == '$' && s[1] == '$' && !s[2]) {
		sprintf(numbuf, "%d", getpid());
		return numbuf;
	} else if (s[0] == 'y' && s[1] == 'r' && !s[2]) {
		if (eregs_idx[r])
			v = env->reg[eregs_idx[r]];
		else
			v = nregs[r];
		sprintf(numbuf, "%02d", v);
		return numbuf;
	}
	d = nregs_fmt[r];
	ei = eregs_idx[r];
	if (!d)
		goto plain;
	if (!ei)
		v = nregs[r];
	else
		v = env->reg[ei];
	if (v < 0) {
		numbuf[0] = '-';
		v = -v;
		p = numbuf + 1;
	} else
		p = numbuf;
	c = d & 0xff;
	if ((d & 0xdf) == 'I' && (unsigned)(v - 1) <= 0x9c3e) {
		if (c == 'i')
			nf_roman(p, v, "ixcmz", "vldw");
		else
			nf_roman(p, v, "IXCMZ", "VLDW");
	} else if ((d & 0xdf) == 'A' && v) {
		q = p;
		do {
			n = v - 1;
			*q++ = n % 26 + d;
			v = n / 26;
		} while (v);
		*q = 0;
	} else if (c == '0' || c == 'x' || c == 'X') {
		if (c == '0')
			c = 'd';
		sprintf(buf, "%%0%d%c", d >> 8, c);
		sprintf(p, buf, v);
		return numbuf;
	} else
		goto plain;
	e = stpcpy(buf, p);
	if (e != buf)
		for (q = p; e != buf; )
			*q++ = *--e;
	return numbuf;
plain:
	sprintf(numbuf, "%d", ei ? env->reg[ei] : nregs[r]);
	return numbuf;
}
