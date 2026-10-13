#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "fn_sbuf.h"
#include "fn_sbuf_str.h"
#include "fn_mbox_open.h"
#include "fn_mbox_len.h"
#include "fn_mbox_get.h"
#include "fn_mbox_ith.h"
#include "fn_mbox_free.h"
#include "fn_mbox_off.h"
#include "fn_msg_get.h"
#include "fn_msg_demime.h"
#include "fn_hdrlen.h"
#include "fn_xwrite.h"
#include "fileread.h"
#include "put_id.h"
#include "put_from_.h"

int pg(char **argv)
{
	char *boxes[16] = {0};
	char *hdrs[128] = {0};
	int nbox = 0, nhdr = 0;
	char *addr = NULL;
	int mime = 0, reply = 0, fwd = 0, newm = 0, src = 0;
	int i, n = -1;
	char *msg = NULL;
	long len = 0;
	char *arg;

	for (i = 1; argv[i] && argv[i][0] == '-'; i++) {
		arg = argv[i];
		switch (arg[1]) {
		case 'm':
			mime = 1;
			break;
		case 'r':
			reply = 1;
			break;
		case 'n':
			newm = 1;
			break;
		case 'f':
			fwd = 1;
			break;
		case 's':
			src = 1;
			break;
		case 'b':
		case 'h':
		case 'a':
		case 'i': {
			char *v = arg[2] ? arg + 2 : argv[++i];
			if (!v)
				break;
			if (arg[1] == 'b')
				boxes[nbox++] = v;
			else if (arg[1] == 'h') {
				if (nhdr < 127)
					hdrs[nhdr++] = v;
			} else if (arg[1] == 'a') {
				if (parts_n <= 15)
					parts[parts_n++] = v;
			} else
				addr = v;
			break;
		}
		}
	}
	if (newm) {
		struct sbuf *sb = sbuf_make();
		time_t now;
		char buf[0x80];
		put_from_(sb);
		sbuf_printf(sb, "From: \n");
		sbuf_printf(sb, "To: \n");
		sbuf_printf(sb, "Subject: \n");
		put_id(sb);
		now = time(NULL);
		strftime(buf, sizeof(buf), "%a, %d %b %Y %H:%M:%S %z", localtime(&now));
		sbuf_printf(sb, "Date: %s\n", buf);
		sbuf_printf(sb, "User-Agent: Neatmail/1.1 (https://github.com/aligrudi/neatmail)\n");
		sbuf_printf(sb, "MAIL BODY...\n");
		put_body(sb, "MAIL BODY...\n");
		sbuf_chr(sb, '\n');
		len = sbuf_len(sb);
		msg = sbuf_done(sb);
		xwrite(1, msg, len);
		free(msg);
		return 0;
	}
	if (!nbox && argv[i])
		boxes[nbox++] = argv[i++];
	if (!addr && argv[i])
		addr = argv[i];
	if (!boxes[0] || !addr) {
		puts("usage: neatmail pg [options] [address]\n\noptions:\n"
		     "   -h hdr  \tspecify headers to include\n"
		     "   -m      \tdecode mime message\n"
		     "   -r      \tgenerate a reply\n"
		     "   -f      \tgenerate a forward\n"
		     "   -n      \tgenerate a new message\n"
		     "   -a file \tadd an attachment\n"
		     "   -s      \tfollow neatmail-source: header\n\n"
		     "address:\n"
		     "   mbox msgnum \tpage msgnum-th message in mbox\n"
		     "   mbox =msgid \tpage the message with the given message-id\n"
		     "   msgnum@mbox \tsame as mbox msgnum");
		return 1;
	}
	{
		char *at = strchr(addr, '@');
		if (at) {
			*at = 0;
			boxes[0] = at + 1;
		}
	}
	if (addr[0] == '=') {
		struct mbox *m = mbox_open(boxes[0]);
		char *id = addr + 1;
		int k, total;
		if (!m)
			return 1;
		total = mbox_len(m);
		for (k = 0; k < total; k++) {
			char *s, *p, *lt, *gt;
			long sl;
			if (mbox_get(m, k, &s, &sl))
				continue;
			p = msg_get(s, sl, "Message-ID:");
			if (!p)
				continue;
			{
				int hl = hdrlen(p, s + sl - p);
				lt = memchr(p, '<', hl);
				if (!lt)
					continue;
				gt = memchr(p, '>', hl);
				if (gt < lt)
					continue;
				if ((long)(gt - lt - 1) == (long)strlen(id) && !memcmp(lt + 1, id, strlen(id)))
					n = k;
			}
		}
		mbox_free(m);
	} else {
		n = strtol(addr, NULL, 10);
	}
	if (n < 0)
		return 1;
	if (mbox_ith(boxes[0], n, &msg, &len))
		return 1;
	if (src) {
		char *hdr = msg_get(msg, len, "Neatmail-Source:");
		char name[0x100];
		int num = 0;
		long off = 0, l = 0;
		if (hdr) {
			while (*hdr == ' ' || *hdr == '\t')
				hdr++;
			if (sscanf(hdr, "%d@%s %ld %ld", &num, name, &off, &l) == 4) {
				char *nm = NULL;
				long nl = 0;
				free(msg);
				if (!mbox_off(name, off, l, &nm, &nl)) {
					msg = nm;
					len = nl;
				}
			}
		}
	}
	if (mime) {
		char *out = NULL;
		long outn = 0;
		if (!msg_demime(msg, len, &out, &outn)) {
			free(msg);
			msg = out;
			len = outn;
		}
	}
	if (reply) {
		struct sbuf *sb = sbuf_make();
		char *from = msg_get(msg, len, "From:");
		char *subj = msg_get(msg, len, "Subject:");
		char *mid = msg_get(msg, len, "Message-ID:");
		char *refs = msg_get(msg, len, "References:");
		char *to = msg_get(msg, len, "To:");
		char *cc = msg_get(msg, len, "CC:");
		char *rto = msg_get(msg, len, "Reply-To:");
		char *dst = rto ? rto : from;
		time_t now;
		char buf[0x80];
		put_from_(sb);
		now = time(NULL);
		strftime(buf, sizeof(buf), "%a, %d %b %Y %H:%M:%S %z", localtime(&now));
		sbuf_printf(sb, "Date: %s\n", buf);
		sbuf_printf(sb, "From: \n");
		if (dst) {
			char *c = strchr(dst, ':');
			char *v = c ? c + 1 : dst;
			int vl;
			while (*v == ' ')
				v++;
			vl = hdrlen(v, 0x400);
			sbuf_str(sb, "To: ");
			sbuf_mem(sb, v, vl);
			sbuf_str(sb, "\n");
		}
		(void)to;
		(void)cc;
		if (subj) {
			char *c = strchr(subj, ':');
			char *v = c ? c + 1 : subj;
			while (isspace((unsigned char)*v))
				v++;
			sbuf_str(sb, "Subject: ");
			if (!(tolower((unsigned char)v[0]) == 'r' && tolower((unsigned char)v[1]) == 'e'))
				sbuf_str(sb, "Re: ");
			sbuf_mem(sb, v, hdrlen(v, 0x400));
			sbuf_str(sb, "\n");
		}
		put_id(sb);
		if (mid) {
			char *c = strchr(mid, ':');
			char *v = c ? c + 1 : mid;
			while (isspace((unsigned char)*v))
				v++;
			sbuf_str(sb, "In-Reply-To: ");
			sbuf_mem(sb, v, hdrlen(v, 0x400));
			sbuf_str(sb, "\n");
		}
		if (refs) {
			char *c = strchr(refs, ':');
			char *v = c ? c + 1 : refs;
			while (isspace((unsigned char)*v))
				v++;
			sbuf_str(sb, "References: ");
			sbuf_mem(sb, v, hdrlen(v, 0x400));
			sbuf_str(sb, "\n\t");
			if (mid)
				sbuf_str(sb, mid);
			sbuf_str(sb, "\n");
		}
		sbuf_printf(sb, "User-Agent: Neatmail/1.1 (https://github.com/aligrudi/neatmail)\n");
		{
			char *body = msg;
			char *end = msg + len;
			char *q = msg;
			struct sbuf *qb = sbuf_make();
			char *qs;
			while (q < end && !(q[-1] == '\n' && *q == '\n' && q > msg))
				q++;
			body = q;
			sbuf_str(qb, "\n");
			(void)body;
			qs = sbuf_buf(qb);
			(void)qs;
			sbuf_free(qb);
			sbuf_str(sb, "\n");
		}
		put_body(sb, msg);
		len = sbuf_len(sb);
		free(msg);
		msg = sbuf_done(sb);
		xwrite(1, msg, len);
		free(msg);
		return 0;
	}
	if (nhdr) {
		struct sbuf *sb = sbuf_make();
		char *end = msg + len;
		char *q;
		int k;
		for (k = 0; k < nhdr && hdrs[k]; k++) {
			char *p = msg_get(msg, len, hdrs[k]);
			if (p)
				sbuf_mem(sb, p, hdrlen(p, end - p));
		}
		q = msg + 1;
		while (q < end && !(q[-1] == '\n' && *q == '\n'))
			q++;
		sbuf_mem(sb, q, end - q);
		len = sbuf_len(sb);
		free(msg);
		msg = sbuf_done(sb);
	}
	if (fwd) {
		struct sbuf *sb = sbuf_make();
		struct sbuf *sbo = sbuf_make();
		char *subj = msg_get(msg, len, "Subject:");
		time_t now;
		char buf[0x80];
		put_from_(sb);
		now = time(NULL);
		strftime(buf, sizeof(buf), "%a, %d %b %Y %H:%M:%S %z", localtime(&now));
		sbuf_printf(sb, "Date: %s\n", buf);
		sbuf_printf(sb, "From: \n");
		sbuf_printf(sb, "To: \n");
		if (subj) {
			char *c = strchr(subj, ':');
			char *v = c + 1;
			while (isspace((unsigned char)*v))
				v++;
			sbuf_str(sb, "Subject: ");
			sbuf_str(sb, "Fwd: ");
			sbuf_mem(sb, v, hdrlen(v, 0x400));
			sbuf_str(sb, "\n");
		} else {
			sbuf_str(sb, "Subject: ");
			sbuf_str(sb, "Fwd: ");
			sbuf_str(sb, "\n");
		}
		put_id(sb);
		sbuf_printf(sb, "User-Agent: Neatmail/1.1 (https://github.com/aligrudi/neatmail)\n");
		sbuf_str(sbo, "\n-------- Original Message --------\n");
		sbuf_mem(sbo, msg, len);
		put_body(sb, sbuf_buf(sbo));
		sbuf_free(sbo);
		len = sbuf_len(sb);
		free(msg);
		msg = sbuf_done(sb);
	}
	xwrite(1, msg, len);
	free(msg);
	return 0;
}
