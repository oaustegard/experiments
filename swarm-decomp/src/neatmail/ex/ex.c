#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fn_mbox_free.h"
#include "fn_mbox_get.h"
#include "fn_mbox_len.h"
#include "fn_mbox_open.h"
#include "fn_mbox_set.h"
#include "fn_msg_get.h"
#include "fn_msg_set.h"
#include "fn_regfree.h"
#include "vars.h"
#include "ex_lineno.h"
#include "ex_exec.h"

int ex(char **argv)
{
	char *av[16];
	char buf[512];
	char cmd[512];
	char *p, *s, *t;
	int c;
	int j, k, i, lo, hi;
	char st = 0;
	int ch2;
	char *ms, *out;
	long mn, outlen;
	char tmp[16];

	memset(av, 0, sizeof av);
	j = 0;
	k = 0;
	if (!argv[0]) {
		puts("usage: neatmail ex [mbox] <cmds");
		return 1;
	}
	while (argv[j]) {
		if (argv[j][0] != '-') {
			while (argv[j])
				av[k++] = argv[j++];
			break;
		}
		if (argv[j][1] == 'b') {
			char *a;
			if (argv[j][2])
				a = argv[j] + 2;
			else
				a = argv[++j];
			av[k++] = a;
		}
		j++;
	}
	mbox = mbox_open(av);
	if (!mbox) {
		fprintf(stderr, "neatmail: cannot open <%s>\n", av[0]);
		return 1;
	}
	while (fgets(buf, 512, stdin)) {
		ch2 = buf[1];
		if (isupper((unsigned char)buf[0]) && ch2) {
			char *q, *end;
			int start, num, r, st2;
			if (isupper((unsigned char)buf[0]))
				st = buf[0];
			if ((unsigned char)(ch2 - 'A') < 26) {
				start = 2;
			} else {
				ch2 = 0;
				start = 1;
			}
			q = buf + start;
			num = 0;
			if ((unsigned char)(*q - '0') <= 9) {
				do {
					num = num * 10 + (*q - '0');
					q++;
				} while ((unsigned char)(*q - '0') <= 9);
			}
			if (*q != '@') {
				r = mbox_get(mbox, num, &ms, &mn);
				if (r == 0) {
					char *ps, *r8;
					char nx;
					pos = num;
					ps = msg_get(ms, mn, "status:");
					st2 = 'N';
					if (ps) {
						q = ps + 7;
						end = ms + mn;
						do {
							r8 = q;
							c = *q++;
						} while (q < end && isspace((unsigned char)c));
						st2 = isspace(c) ? 'N' : c;
						nx = r8[1];
						if (!isspace(nx))
							r = (unsigned char)nx;
					}
					if (st2 != st || r != ch2) {
						if (ch2 == 0)
							sprintf(tmp, "Status: %c\n", st);
						else
							sprintf(tmp, "Status: %c%c\n", st, ch2);
						if (msg_set(ms, mn, &out, &outlen, "status:", tmp) == 0) {
							mbox_set(mbox, pos, out, outlen);
							free(out);
						}
					}
				}
			}
		}
		if (buf[0] != ':')
			continue;
		p = buf;
		do {
			c = *++p;
		} while (c == ':' || isspace((unsigned char)c));
		s = cmd;
		while (*p && !isalpha((unsigned char)*p)) {
			if (*p == '/' || *p == '?') {
				c = *p;
				*s++ = *p++;
				while (*p && *p != c) {
					if (*p == '\\' && p[1])
						*s++ = *p++;
					*s++ = *p++;
				}
				if (*p)
					*s++ = *p++;
				continue;
			}
			*s++ = *p++;
		}
		*s = 0;
		t = cmd;
		for (;;) {
			int ok = 1;
			if (*t == '%') {
				lo = 0;
				hi = mbox_len(mbox);
			} else if (*t == 0 || *t == ';') {
				lo = pos;
				hi = pos + (pos != mbox_len(mbox));
			} else {
				lo = ex_lineno(t);
				s = t;
				while (*s && *s != ',' && *s != ';')
					s++;
				if (*s == ',') {
					hi = ex_lineno(s + 1) + 1;
				} else {
					hi = mbox_len(mbox);
					if (lo != hi)
						hi = lo + 1;
				}
				if (lo < 0 || lo >= mbox_len(mbox) || lo > hi || mbox_len(mbox) < hi)
					ok = 0;
			}
			if (ok) {
				for (i = lo; i < hi; i++) {
					pos = i;
					ex_exec(p);
				}
			}
			while (*t && *t != ';')
				t++;
			if (*t == ';')
				t++;
			if (!*t)
				break;
		}
	}
	mbox_free(mbox);
	if (kwd[0])
		regfree(&kwd_re);
	return 0;
}
