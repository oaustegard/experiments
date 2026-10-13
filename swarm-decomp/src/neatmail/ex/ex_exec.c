#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fn_mbox_get.h"
#include "fn_mbox_set.h"
#include "fn_mbox_copy.h"
#include "fn_mbox_save.h"
#include "fn_msg_set.h"
#include "fn_msg_get.h"
#include "fn_hdrlen.h"
#include "fn_sbuf.h"
#include "fn_xpipe.h"
#include "vars.h"
#include "ex_arg.h"
#include "ec_cp.h"
#include "ec_g.h"
#include "ex_exec.h"

static int ex_exec(char *s)
{
	char cmd[512], a1[512], a2[512], a3[512];
	char *p, *q, *s1, *c, *out, *end, *val;
	long n, len, outn;
	struct sbuf *sb;
	int i, j, k, hl;

	while (isspace(*s))
		s++;
	i = 0;
	while (isalpha(*s))
		cmd[i++] = *s++;
	cmd[i] = 0;
	if (!strcmp(cmd, "rm")) {
		mbox_set(mbox, pos, "", 0);
		return 0;
	}
	if (!strcmp(cmd, "cp"))
		return ec_cp(s);
	if (!strcmp(cmd, "mv")) {
		ec_cp(s);
		mbox_set(mbox, pos, "", 0);
		return 0;
	}
	if (!strcmp(cmd, "hd") || !strcmp(cmd, "set")) {
		s = ex_arg(s, a1);
		if (mbox_get(mbox, pos, &p, &len))
			return 0;
		a3[0] = 0;
		while (isspace(*s))
			s++;
		if (*s) {
			ex_arg(s, a2);
			snprintf(a3, 512, "%s %s\n", a1, a2);
		}
		if (msg_set(p, len, &out, &outn, a1, a3))
			return 0;
		mbox_set(mbox, pos, out, outn);
		free(out);
		return 0;
	}
	if (!strcmp(cmd, "ft") || !strcmp(cmd, "file")) {
		ex_arg(s, a3);
		if (mbox_get(mbox, pos, &p, &len))
			return 0;
		if (xpipe(a3, p, len, &out, &outn))
			return 0;
		mbox_set(mbox, pos, out, outn);
		free(out);
		return 0;
	}
	if (!strcmp(cmd, "w")) {
		ex_arg(s, a3);
		if (a3[0] == 0)
			mbox_save(mbox);
		else
			mbox_copy(mbox, a3);
		return 0;
	}
	if (!strcmp(cmd, "g"))
		return ec_g(s, 0);
	if (!strcmp(cmd, "g!"))
		return ec_g(s, 1);
	if (!strcmp(cmd, "tj")) {
		n = strtol(s, 0, 10);
		if (mbox_get(mbox, n, &s1, &len))
			return 0;
		if (mbox_get(mbox, pos, &p, &outn))
			return 0;
		q = msg_get(s1, len, "message-id:");
		if (!q)
			return 0;
		hl = hdrlen(q, s1 + len - q);
		end = q + hl;
		c = strchr(q, ':');
		sb = sbuf_make();
		sbuf_str(sb, "In-Reply-To:");
		sbuf_mem(sb, c + 1, end - (c + 1));
		val = sbuf_done(sb);
		if (msg_set(p, outn, &out, &len, "in-reply-to:", val))
			return 0;
		mbox_set(mbox, pos, out, len);
		free(out);
		return 0;
	}
	if (!strcmp(cmd, "ch") || !strcmp(cmd, "chop")) {
		n = strtol(s, 0, 10);
		if (mbox_get(mbox, pos, &p, &len))
			return 0;
		j = 0;
		if (len > 2) {
			j = len - 2;
			for (k = 0; k != j; k++)
				if (p[k] == '\n' && p[k + 1] == '\n')
					j = k;
		}
		k = (int)(n * 1024) + j + 1;
		if (k >= len)
			return 0;
		while (p[k] != '\n')
			if (++k == len)
				return 0;
		sb = sbuf_make();
		sbuf_mem(sb, p, k);
		sbuf_str(sb, "\nCHOPPED...\n\n");
		outn = sbuf_len(sb);
		out = sbuf_done(sb);
		mbox_set(mbox, pos, out, outn);
		free(out);
		return 0;
	}
	if (!strcmp(cmd, "p"))
		return printf("%d\n", pos);
	return 0;
}
