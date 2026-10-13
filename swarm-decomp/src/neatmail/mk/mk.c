#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
int datedate(char *s);
#include "fn_mbox_free.h"
#include "fn_mbox_get.h"
#include "fn_mbox_len.h"
#include "fn_mbox_open.h"
#include "fn_mbox_pos.h"
#include "fn_hdrlen.h"
#include "fn_msg_get.h"
#include "fn_msg_hdrdec.h"
#include "fn_sbuf.h"
#include "fn_startswith.h"
#include "mkmsg.h"

int mk(char **argv) {
	struct mbox *m;
	struct mkmsg *msgs, *x;
	struct mkmsg **bydate, **byid, **ord;
	char *fmt[10];
	char *files[64];
	char buf[256], pre[64], key[64];
	struct tm *lt;
	time_t t;
	char *v;
	int w;
	char *s, *p, *q;
	long len;
	int i, j, k, nfiles = 0, nmsg, start = 0, wn = 0, wm = 4;
	int summary = 0, sort = 0, pos, f, c1, c2, cnt[64];
	int *ia, *ib;

	memset(fmt, 0, sizeof fmt);
	memset(files, 0, sizeof files);
	if (!argv[0])
		goto usage;
	for (i = 0; argv[i];) {
		if (argv[i][0] != '-') {
			for (nfiles = 0; argv[i]; i++)
				files[nfiles++] = argv[i];
			break;
		}
		switch (argv[i][1]) {
		case '0': case '1': case '2': case '3': case '4':
		case '5': case '6': case '7': case '8': case '9':
			if (argv[i][2]) {
				fmt[argv[i][1] - '0'] = argv[i] + 2;
			} else {
				i++;
				fmt[argv[i - 1][1] - '0'] = argv[i];
			}
			break;
		case 'f':
			p = argv[i][2] ? argv[i] + 2 : argv[++i];
			start = strtol(p, NULL, 10);
			break;
		case 'n':
			p = argv[i][2] ? argv[i] + 2 : argv[++i];
			wn = strtol(p, NULL, 10);
			break;
		case 'm':
			p = argv[i][2] ? argv[i] + 2 : argv[++i];
			wm = strtol(p, NULL, 10);
			break;
		case 'r':
			summary = 1;
			break;
		case 's':
			p = argv[i][2] ? argv[i] + 2 : argv[++i];
			sort = (p[0] == 't') + 1;
			break;
		}
		i++;
	}
	if (!files[0])
		goto usage;

	m = mbox_open(files);
	if (!m) {
		fprintf(stderr, "neatmail: cannot open <%s>\n", files[0]);
		return 1;
	}
	nmsg = mbox_len(m);
	msgs = calloc(nmsg, sizeof *msgs);
	bydate = malloc(nmsg * sizeof *bydate);
	byid = malloc(nmsg * sizeof *byid);
	ord = malloc(nmsg * sizeof *ord);
	if (!msgs || !bydate || !byid || !ord)
		return 1;
	for (i = 0; i < nmsg; i++) {
		x = &msgs[i];
		x->idx = i;
		mbox_get(m, i, &x->s, &x->n);
		bydate[i] = byid[i] = x;
		p = msg_get(x->s, x->n, "Message-ID:");
		if (p) {
			q = p + hdrlen(p, x->s + x->n - p);
			(void)q;
		}
	}
	qsort(bydate, nmsg, sizeof *bydate, msgcmp_date);
	qsort(byid, nmsg, sizeof *byid, msgcmp_id);
	for (i = 0; i < nmsg; i++) {
		x = bydate[i];
		p = msg_get(x->s, x->n, "Date:");
		if (p)
			x->date = datedate(p + 5);
		p = msg_get(x->s, x->n, "In-Reply-To:");
		if (p) {
			x->irt = p;
			x->irtlen = hdrlen(p, x->s + x->n - p);
		}
	}
	for (i = 0; i < nmsg; i++) {
		x = &msgs[i];
		if (x->irt) {
			for (j = 0; j < nmsg; j++)
				if (x->irtlen == msgs[j].idlen && strncmp(x->irt, msgs[j].id, x->irtlen) == 0) {
					x->parent = &msgs[j];
					break;
				}
		}
	}
	for (i = 0; i < nmsg; i++) {
		x = &msgs[i];
		if (x->parent) {
			x->depth = x->parent->depth + 1;
			if (!x->parent->child)
				x->parent->child = x;
			else
				x->parent->lastchild->next = x;
			x->parent->lastchild = x;
		}
	}
	if (nmsg > 0 && !sort) {
		struct mkmsg **out = ord;
		for (i = 0; i < nmsg; i++) {
			x = &msgs[i];
			if (!x->parent)
				out = put_msg(out, x);
		}
	}
	ia = malloc(nmsg * sizeof *ia);
	ib = malloc(nmsg * sizeof *ib);
	for (i = 0; i < nmsg; i++) {
		ia[i] = i;
		ib[i] = 0;
	}
	if (summary) {
		memset(cnt, 0, sizeof cnt);
		for (i = 0; i < nmsg; i++) {
			mbox_get(m, i, &s, &len);
			c1 = msg_stat(s, len, 0, 'N');
			cnt[(c1 - 'A') & 63]++;
		}
		for (i = 0; i < 26; i++)
			if (cnt[i])
				fprintf(stderr, "%c%04d ", 'A' + i, cnt[i]);
		fputc('\n', stderr);
		goto done;
	}
	if (sort) {
		for (j = 0; j < nmsg; j++)
			ord[j] = bydate[j];
	} else {
		for (j = 0; j < nmsg; j++)
			ord[j] = &msgs[j];
	}
	for (i = start; i < nmsg; i++) {
		x = ord[i];
		mbox_get(m, x->idx, &s, &len);
		c1 = msg_stat(s, len, 1, '0');
		c2 = msg_stat(s, len, 0, 'N');
		f = mbox_pos(m, x->idx, &pos);
		snprintf(pre, sizeof pre, "%c%c%*d %*d", c2, c1, wm, f, wn, pos);
		for (k = 0; k < 10 && fmt[k]; k++) {
			printf("%s ", pre);
			for (p = fmt[k]; *p;) {
				w = 0;
				while (isdigit((unsigned char)*p))
					w = w * 10 + *p++ - '0';
				for (q = key; *p && *p != ':';)
					*q++ = *p++;
				if (*p == ':')
					*q++ = *p++;
				*q = 0;
				if (strcmp(key, "~date:") == 0) {
					t = datedate(msg_get(s, len, "Date:") + 5);
					lt = localtime(&t);
					strftime(buf, sizeof buf, "%d %b %Y %H:%M:%S", lt);
					printf("%*s", w, buf);
				} else {
					v = msg_get(s, len, key);
					if (v)
						printf("%*.*s", w, hdrlen(v, len - (v - s)), v);
					else
						printf("%*s", w, "");
				}
			}
			putchar('\n');
		}
	}
done:
	free(ia);
	free(ib);
	free(msgs);
	free(bydate);
	free(byid);
	free(ord);
	mbox_free(m);
	return 0;
usage:
	puts("usage: neatmail mk [options] [mbox]\n\noptions:\n"
	     "   -0 fmt  \tmessage first line format (e.g., 20from:40subject:)\n"
	     "   -1 fmt  \tmessage second line format\n"
	     "   -sd     \tsort by receiving date\n"
	     "   -st     \tsort by threads\n"
	     "   -r      \tprint a summary of status flags\n"
	     "   -f n    \tthe first message to list\n"
	     "   -n n    \tmessage index field width\n"
	     "   -m n    \tmessage file field width");
	return 1;
}
