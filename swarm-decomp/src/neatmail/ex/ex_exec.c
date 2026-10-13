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

static int ex_exec(char *s)
{
	char cmd[512];
	char *c;

	while (isspace((unsigned char)*s))
		s++;
	c = cmd;
	while (isalpha((unsigned char)*s))
		*c++ = *s++;
	*c = 0;
	if (!strcmp(cmd, "rm")) {
		return mbox_set(mbox, pos, "", 0);
	} else if (!strcmp(cmd, "cp")) {
		return ec_cp(s);
	} else if (!strcmp(cmd, "mv")) {
		ec_cp(s);
		return mbox_set(mbox, pos, "", 0);
	} else if (!strcmp(cmd, "hd") || !strcmp(cmd, "set")) {
		char a1[512], a2[512], a3[512];
		char *p, *out;
		long len, outn;
		char *q;

		s = ex_arg(s, a1);
		if (!mbox_get(mbox, pos, &p, &len)) {
			a3[0] = 0;
			q = s;
			while (isspace((unsigned char)*q))
				q++;
			if (*q) {
				ex_arg(s, a2);
				snprintf(a3, 512, "%s %s\n", a1, a2);
			}
			if (!msg_set(p, len, &out, &outn, a1, a3)) {
				mbox_set(mbox, pos, out, outn);
				free(out);
			}
		}
	} else if (!strcmp(cmd, "ft") || !strcmp(cmd, "filt")) {
		char a3[512];
		char *p, *out;
		long len, outn;

		ex_arg(s, a3);
		if (!mbox_get(mbox, pos, &p, &len)) {
			if (!xpipe(a3, p, len, &out, &outn)) {
				mbox_set(mbox, pos, out, outn);
				free(out);
			}
		}
	} else if (!strcmp(cmd, "w")) {
		char a3[512];

		ex_arg(s, a3);
		if (a3[0] == 0)
			mbox_save(mbox);
		else
			mbox_copy(mbox, a3);
	} else if (!strcmp(cmd, "g")) {
		return ec_g(s, 0);
	} else if (!strcmp(cmd, "g!")) {
		return ec_g(s, 1);
	} else if (!strcmp(cmd, "tj")) {
		char *s1, *p, *q, *end, *c2, *out, *val;
		long n, len, outn, outl;
		int hl;
		struct sbuf *sb;

		n = strtol(s, 0, 10);
		if (!mbox_get(mbox, n, &s1, &len)) {
			if (!mbox_get(mbox, pos, &p, &outn)) {
				q = msg_get(s1, len, "message-id:");
				if (q) {
					hl = hdrlen(q, s1 + len - q);
					end = q + hl;
					c2 = strchr(q, ':');
					sb = sbuf_make();
					sbuf_str(sb, "In-Reply-To:");
					sbuf_mem(sb, c2 + 1, end - (c2 + 1));
					val = sbuf_done(sb);
					if (!msg_set(p, outn, &out, &outl, "in-reply-to:", val)) {
						mbox_set(mbox, pos, out, outl);
						free(out);
					}
				}
			}
		}
	} else if (!strcmp(cmd, "ch") || !strcmp(cmd, "chop")) {
		long n, j, k;
		char *p, *out;
		long len, outn;
		struct sbuf *sb;

		n = strtol(s, 0, 10);
		if (!mbox_get(mbox, pos, &p, &len)) {
			j = 0;
			if (len > 2) {
				j = len - 2;
				for (k = 0; k != j; k++)
					if (p[k] == '\n' && p[k + 1] == '\n')
						j = k;
			}
			k = (int)(n * 1024) + j + 1;
			if (k < len) {
				while (p[k] != '\n')
					if (++k == len)
						break;
				if (k != len) {
					sb = sbuf_make();
					sbuf_mem(sb, p, k);
					sbuf_str(sb, "\nCHOPPED...\n\n");
					outn = sbuf_len(sb);
					out = sbuf_done(sb);
					mbox_set(mbox, pos, out, outn);
					free(out);
				}
			}
		}
	} else if (!strcmp(cmd, "p")) {
		return printf("%d\n", pos);
	}
}
