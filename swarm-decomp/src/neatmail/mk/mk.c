#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct _IO_FILE;
extern struct _IO_FILE *stderr;
int printf(const char *, ...);
int fprintf(struct _IO_FILE *, const char *, ...);
int fputc(int, struct _IO_FILE *);
int putchar(int);
int puts(const char *);
int sprintf(char *, const char *, ...);
int snprintf(char *, unsigned long, const char *, ...);

int mk(char **argv) {
	static char num[32];
	struct mbox *m;
	struct mkmsg *msgs, *x;
	struct mkmsg **bydate, **byid, **ord;
	char *fmt[4];
	char *files[16];
	char fb0[32], fb1[32];
	int i, j, k, nfiles = 0, nmsg, start = 0, mw = 0, nw = 4, summary = 0, sort = 0;
	int *ia, *ib;
	int pos, f, c1, c2, depth, col, w;
	char *p, *q, *s, *t, *v, *d, *end, *buf1, *kp;
	long len;

	memset(fmt, 0, sizeof fmt);
	memset(files, 0, sizeof files);
	fmt[0] = "18from:40~subject:";
	if (!argv[0])
		goto usage;
	for (i = 0; argv[i];) {
		if (argv[i][0] != '-') {
			for (; argv[i]; i++)
				files[nfiles++] = argv[i];
			break;
		}
		switch (argv[i][1]) {
		case 'b':
			p = argv[i][2] ? argv[i] + 2 : argv[++i];
			files[nfiles++] = p;
			break;
		case 'f':
			p = argv[i][2] ? argv[i] + 2 : argv[++i];
			start = strtol(p, NULL, 10);
			break;
		case 'm':
			p = argv[i][2] ? argv[i] + 2 : argv[++i];
			mw = strtol(p, NULL, 10);
			break;
		case 'n':
			p = argv[i][2] ? argv[i] + 2 : argv[++i];
			nw = strtol(p, NULL, 10);
			break;
		case 'r':
			summary = 1;
			break;
		case 's':
			p = argv[i][2] ? argv[i] + 2 : argv[++i];
			sort = (p[0] == 't') + 1;
			break;
		default:
			if (argv[i][1] == '0' || argv[i][1] == '1') {
				int d = argv[i][1] - '0';
				fmt[d] = argv[i][2] ? argv[i] + 2 : argv[++i];
			}
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
	ia = malloc(mbox_len(m) * sizeof *ia);
	ib = malloc(mbox_len(m) * sizeof *ib);
	for (i = 0; i < mbox_len(m); i++)
		ia[i] = i;
	for (i = 0; i < mbox_len(m); i++)
		ib[i] = 0;

	if (sort) {
		int *ib2 = (sort == 2) ? ib : NULL;
		nmsg = mbox_len(m);
		msgs = calloc(nmsg * sizeof *msgs, 1);
		bydate = malloc(nmsg * sizeof *bydate);
		byid = malloc(nmsg * sizeof *byid);
		ord = malloc(nmsg * sizeof *ord);
		if (msgs && bydate && byid) {
			for (i = 0; i < nmsg; i++) {
				x = &msgs[i];
				x->idx = i;
				mbox_get(m, i, &x->s, &x->n);
				bydate[i] = x;
				byid[i] = x;
				end = x->s + x->n;
				p = msg_get(x->s, x->n, "Message-ID:");
				q = msg_get(x->s, x->n, "In-Reply-To:");
				s = msg_get(x->s, x->n, "Date:");
				if (p) {
					int n = hdrlen(p, end - p);
					char *lt = memchr(p, '<', n);
					if (lt) {
						char *gt = memchr(p, '>', n);
						if (gt) {
							while (*lt == '<')
								lt++;
							x->id = lt;
							x->idlen = gt - lt;
						}
					}
				}
				if (q) {
					int n = hdrlen(q, end - q);
					char *lt = memchr(q, '<', n);
					if (lt) {
						char *gt = memchr(q, '>', n);
						if (gt) {
							while (*lt == '<')
								lt++;
							x->irt = lt;
							x->irtlen = gt - lt;
						}
					}
				}
				if (s) {
					x->date = datedate(s + 5);
				} else {
					char rb[128];
					int day, hh, mm, ss, yr, mon = 0, ii;
					int dt;
					p = readtok((unsigned char *)x->s, rb);
					p = readtok((unsigned char *)p, rb);
					p = readtok((unsigned char *)p, rb);
					p = readtok((unsigned char *)p, rb);
					for (ii = 0; ii < 12; ii++)
						if (strcmp(months[ii], rb) == 0)
							mon = ii;
					q = readtok((unsigned char *)p, rb);
					day = strtol(rb, NULL, 10);
					q = readtok((unsigned char *)q, rb);
					hh = strtol(rb, NULL, 10);
					q = readtok((unsigned char *)q, rb);
					mm = strtol(rb, NULL, 10);
					q = readtok((unsigned char *)q, rb);
					ss = strtol(rb, NULL, 10);
					if (ss <= 59)
						readtok((unsigned char *)q, rb);
					yr = strtol(rb, NULL, 10);
					dt = ((yr - 1970) * 400 + mon * 31 + day) * 86400 + hh * 3600 + mm * 60 + ss;
					x->date = dt;
				}
			}
			qsort(bydate, nmsg, sizeof *bydate, msgcmp_date);
			qsort(byid, nmsg, sizeof *byid, msgcmp_id);
			if (ib2) {
				struct mkmsg **wk;
				for (wk = bydate; wk != bydate + nmsg; wk++) {
					x = *wk;
					if (x->irt) {
						int lo = 0, hi = nmsg, mid, c;
						struct mkmsg *y;
						while (lo < hi) {
							mid = (lo + hi) / 2;
							y = byid[mid];
							c = y->idlen - x->irtlen;
							if (c == 0)
								c = strncmp(x->irt, y->id, x->irtlen);
							if (c == 0) {
								if (y->date < x->date) {
									x->parent = y;
									x->depth = y->depth + 1;
									if (!y->child)
										y->child = x;
									else
										y->lastchild->next = x;
									y->lastchild = x;
								}
								break;
							}
							if (c < 0)
								hi = mid;
							else
								lo = mid + 1;
						}
					}
				}
			}
			{
				struct mkmsg **out = ord;
				for (k = 0; k < nmsg; k++) {
					x = bydate[k];
					if (!x->parent) {
						*out++ = x;
						for (x = x->child; x; x = x->next)
							out = put_msg(out, x);
					}
				}
			}
			for (k = 0; k < nmsg; k++)
				ia[k] = ord[k]->idx;
			if (ib2)
				for (k = 0; k < nmsg; k++)
					ib[k] = ord[k]->depth;
		}
		free(msgs);
		free(bydate);
		free(byid);
		free(ord);
	}

	for (i = start; i < mbox_len(m); i++) {
		int idx = ia[i];
		f = mbox_pos(m, idx, &pos);
		mbox_get(m, idx, &s, &len);
		c1 = msg_stat(s, len, 1, '0');
		c2 = msg_stat(s, len, 0, 'N');
		{
			char pre[32] = "";
			char *tag;
			int c;
			if (f > 0 && (tag = files[f]) != NULL) {
				c = '@';
			} else {
				c = ' ';
				tag = "";
			}
			sprintf(fb0, "%c%%-%ds", c, mw);
			if (mw > 0)
				snprintf(pre, sizeof pre, fb0, tag);
			pre[mw + 1] = 0;
			sprintf(fb1, "%c%c%%0%dd%%s", c2, c1, nw);
			snprintf(num, sizeof num, fb1, pos, pre);
			printf("%s", num);
		}
		for (k = 0; fmt[k]; k++) {
			char buf[128];
			char *kb = malloc(strlen(fmt[k]) + 1);
			if (k > 0)
				putchar('\n');
			p = fmt[k];
			for (;;) {
				q = kb;
				while (*p) {
					int ch = *p++;
					*q++ = ch;
					if (ch == ':')
						break;
				}
				*q = 0;
				kp = kb;
				if (!*kp)
					break;
				w = 0;
				while ((unsigned char)(*kp - '0') <= 9)
					w = w * 10 + *kp++ - '0';
				putchar('\t');
				if (strcmp("~subject:", kp) == 0) {
					depth = ib[i];
					if (depth > 0) {
						j = 0;
						do {
							j++;
							if (w < 0 || j < w)
								putchar(' ');
						} while (j != depth);
						if (w > 0)
							w = depth < w ? w - depth : 1;
					}
				}
				v = msg_get(s, len, kp + (*kp == '~'));
				d = NULL;
				if (v) {
					int n = hdrlen(v, s + len - v);
					buf1 = malloc(n);
					memcpy(buf1, v, n - 1);
					buf1[n - 1] = 0;
					d = msg_hdrdec(buf1);
					free(buf1);
				}
				if (d) {
					t = d + strlen(kp) - (*kp == '~');
				} else {
					d = malloc(1);
					*d = 0;
					t = d;
				}
				end = t + strlen(t);
				{
					struct sbuf *sb = sbuf_make();
					char *out;
					while (t < end && isspace((unsigned char)*t))
						t++;
					col = 0;
					if (strcmp("~subject:", kp) == 0) {
						while (startswith(t, "re:") || startswith(t, "fwd:")) {
							sbuf_chr(sb, '+');
							col++;
							t = strchr(t, ':') + 1;
							while (t < end && isspace((unsigned char)*t))
								t++;
						}
					} else if (strcmp("~date:", kp) == 0) {
						time_t tt = datedate(t);
						struct tm *lt = localtime(&tt);
						strftime(buf, sizeof buf, "%d %b %Y %H:%M:%S", lt);
						t = buf;
						end = t + strlen(t);
						(void)buf;
					} else if (strcmp("~size:", kp) == 0) {
						char fb[32];
						sprintf(fb, "%%%dd", w);
						snprintf(buf, sizeof buf, fb, len);
						t = buf;
						end = t + strlen(t);
					}
					while (t < end && (w > col || w <= 0)) {
						int c = (unsigned char)*t;
						int n = 0;
						if ((c & 0xc0) == 0xc0) {
							if (!(c & 0x20))
								n = 2;
							else if (!(c & 0x10))
								n = 3;
							else if (!(c & 0x08))
								n = 4;
						}
						if (n) {
							sbuf_mem(sb, t, n);
							col++;
							t += n;
						} else {
							int ch = (isprint(c) && !isblank(c)) ? c : ' ';
							t++;
							col++;
							sbuf_chr(sb, ch);
						}
					}
					if (w > 0)
						for (; col < w; col++)
							sbuf_chr(sb, ' ');
					free(d);
					out = sbuf_done(sb);
					printf("[%s]", out);
					free(out);
				}
			}
			free(kb);
		}
		putchar('\n');
	}

	free(ia);
	free(ib);
	if (summary) {
		int cnt[128];
		memset(cnt, 0, sizeof cnt);
		for (i = 0; i < mbox_len(m); i++) {
			mbox_get(m, i, &s, &len);
			c1 = msg_stat(s, len, 0, 'N');
			if ((unsigned)(c1 - 'A') <= 25)
				cnt[c1 - 'A']++;
		}
		for (i = 0; i < 128; i++)
			if (cnt[i])
				fprintf(stderr, "%c%04d ", i + 'A', cnt[i]);
		fputc('\n', stderr);
	}
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
