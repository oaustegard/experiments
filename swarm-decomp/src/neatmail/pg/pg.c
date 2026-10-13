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

#define NEATMAIL_USAGE "usage: neatmail pg [options] [address]\n\noptions:\n" \
	"   -h hdr  \tspecify headers to include\n" \
	"   -m      \tdecode mime message\n" \
	"   -r      \tgenerate a reply\n" \
	"   -f      \tgenerate a forward\n" \
	"   -n      \tgenerate a new message\n" \
	"   -a file \tadd an attachment\n" \
	"   -s      \tfollow neatmail-source: header\n\n" \
	"address:\n" \
	"   mbox msgnum \tpage msgnum-th message in mbox\n" \
	"   mbox =msgid \tpage the message with the given message-id\n" \
	"   msgnum@mbox \tsame as mbox msgnum"

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

	for (i = 0; argv[i] && argv[i][0] == '-'; i++) {
		if (argv[i][1] == 'm')
			mime = 1;
		else if (argv[i][1] == 'r')
			reply = 1;
		else if (argv[i][1] == 'n')
			newm = 1;
		else if (argv[i][1] == 'f')
			fwd = 1;
		else if (argv[i][1] == 's')
			src = 1;
		else if (argv[i][1] == 'b') {
			char *v = argv[i][2] ? argv[i] + 2 : argv[++i];
			boxes[nbox++] = v;
		} else if (argv[i][1] == 'i') {
			addr = argv[i][2] ? argv[i] + 2 : argv[++i];
		} else if (argv[i][1] == 'h') {
			char *v = argv[i][2] ? argv[i] + 2 : argv[++i];
			if (nhdr < 127)
				hdrs[nhdr++] = v;
		} else if (argv[i][1] == 'a') {
			char *v = argv[i][2] ? argv[i] + 2 : argv[++i];
			if (parts_n <= 15)
				parts[parts_n++] = v;
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
		time(&now);
		strftime(buf, sizeof(buf), "%a, %d %b %Y %H:%M:%S %z", localtime(&now));
		sbuf_printf(sb, "Date: %s\n", buf);
		sbuf_printf(sb, "User-Agent: Neatmail/1.1 (https://github.com/aligrudi/neatmail)\n");
		put_body(sb, "MAIL BODY...\n");
		sbuf_chr(sb, '\n');
		len = sbuf_len(sb);
		msg = sbuf_done(sb);
		xwrite(1, msg, len);
		free(msg);
		return 0;
	}
	if (!boxes[0] && argv[i])
		boxes[0] = argv[i++];
	if (!addr && argv[i])
		addr = argv[i];
	if (!addr)
		goto usage;
	if (addr[0] == '=') {
		struct mbox *m;
		char *id = addr + 1;
		int k;
		if (!boxes[0])
			goto usage;
		m = mbox_open(boxes);
		if (!m)
			return 1;
		for (k = 0; k < mbox_len(m); k++) {
			char *s = NULL, *p, *lt, *gt;
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
		char *at = strchr(addr, '@');
		if (at) {
			*at = 0;
			boxes[0] = at + 1;
		}
		if (!boxes[0])
			goto usage;
		n = strtol(addr, NULL, 10);
	}
	if (n < 0)
		return 1;
	if (mbox_ith(boxes[0], n, &msg, &len))
		return 1;
	if (src) {
		char *hdr = msg_get(msg, len, "Neatmail-Source:");
		char name[0x200];
		int num = 0;
		long off = 0, l = 0;
		if (hdr) {
			while (*hdr && *hdr != ' ')
				hdr++;
			if (sscanf(hdr, "%d@%s %ld %ld", &num, name, &off, &l) != 4)
				return 1;
			free(msg);
			if (mbox_off(name, off, l, &msg, &len))
				return 1;
		}
	}
	if (mime) {
		char *out;
		long outn;
		if (!msg_demime(msg, len, &out, &outn)) {
			free(msg);
			msg = out;
			len = outn;
		}
	}
	if (reply) {
		struct sbuf *sb = sbuf_make();
		char *mid = msg_get(msg, len, "Message-ID:");
		char *refs = msg_get(msg, len, "References:");
		char *from = msg_get(msg, len, "From:");
		char *subj = msg_get(msg, len, "Subject:");
		char *to = msg_get(msg, len, "To:");
		char *cc = msg_get(msg, len, "CC:");
		char *rto = msg_get(msg, len, "Reply-To:");
		char *dst = rto ? rto : from;
		time_t now;
		char buf[0x80];
		put_from_(sb);
		time(&now);
		strftime(buf, sizeof(buf), "%a, %d %b %Y %H:%M:%S %z", localtime(&now));
		sbuf_printf(sb, "Date: %s\n", buf);
		sbuf_printf(sb, "From: \n");
		if (dst) {
			char *v = strchr(dst, ':') + 1;
			int vl;
			while (isspace(*v))
				v++;
			vl = pg_vlen(v);
			sbuf_str(sb, "To: ");
			sbuf_mem(sb, v, vl);
			sbuf_str(sb, "\n");
		}
		if (to || cc) {
			int cnt = 0;
			sbuf_str(sb, "Cc: ");
			if (to) {
				char *v = strchr(to, ':') + 1;
				int vl;
				while (isspace(*v))
					v++;
				vl = pg_vlen(v);
				sbuf_mem(sb, v, vl);
				cnt = 1;
			}
			if (from && rto) {
				char *v = strchr(from, ':') + 1;
				int vl;
				if (cnt)
					sbuf_str(sb, ",\n\t");
				cnt++;
				while (isspace(*v))
					v++;
				vl = pg_vlen(v);
				sbuf_mem(sb, v, vl);
			}
			if (cc) {
				char *v = strchr(cc, ':') + 1;
				int vl;
				while (isspace(*v))
					v++;
				vl = pg_vlen(v);
				if (cnt)
					sbuf_str(sb, ",\n\t");
				sbuf_mem(sb, v, vl);
			}
			sbuf_str(sb, "\n");
		}
		if (subj) {
			char *v = strchr(subj, ':') + 1;
			while (isspace(*v))
				v++;
			sbuf_str(sb, "Subject: ");
			if (!(tolower(v[0]) == 'r' && tolower(v[1]) == 'e'))
				sbuf_str(sb, "Re: ");
			sbuf_mem(sb, v, pg_vlen(v));
			sbuf_str(sb, "\n");
		}
		put_id(sb);
		if (mid) {
			char *v = strchr(mid, ':') + 1;
			while (isspace(*v))
				v++;
			sbuf_str(sb, "In-Reply-To: ");
			sbuf_mem(sb, v, pg_vlen(v));
			sbuf_str(sb, "\n");
			sbuf_str(sb, "References: ");
			if (refs) {
				char *r = strchr(refs, ':') + 1;
				while (isspace(*r))
					r++;
				sbuf_mem(sb, r, pg_vlen(r));
				sbuf_str(sb, "\n\t");
			}
			sbuf_mem(sb, v, pg_vlen(v));
			sbuf_str(sb, "\n");
		}
		sbuf_printf(sb, "User-Agent: Neatmail/1.1 (https://github.com/aligrudi/neatmail)\n");
		{
			struct sbuf *qb = sbuf_make();
			char *qfrom = msg_get(msg, len, "From:");
			char *end = msg + len;
			char *body = msg + 1;
			while (body < end && !(body[-1] == '\n' && *body == '\n'))
				body++;
			body++;
			sbuf_chr(qb, '\n');
			if (qfrom) {
				char *v = strchr(qfrom, ':') + 1;
				while (isspace(*v))
					v++;
				sbuf_mem(qb, v, pg_vlen(v));
				sbuf_str(qb, " wrote:\n");
			}
			while (body < end) {
				char *nl = memchr(body, '\n', end - body);
				if (!nl)
					break;
				sbuf_str(qb, "> ");
				sbuf_mem(qb, body, nl - body + 1);
				body = nl + 1;
			}
			{
				char *q = sbuf_done(qb);
				put_body(sb, q);
				free(q);
			}
		}
		{
			long nl = sbuf_len(sb);
			char *nm = sbuf_done(sb);
			free(msg);
			msg = nm;
			len = nl;
		}
	}
	if (nhdr) {
		struct sbuf *sb = sbuf_make();
		char *end = msg + len;
		char *q;
		int k;
		for (k = 0; hdrs[k]; k++) {
			char *p = msg_get(msg, len, hdrs[k]);
			if (p)
				sbuf_mem(sb, p, hdrlen(p, end - p));
		}
		q = msg + 1;
		while (q < end && !(q[-1] == '\n' && *q == '\n'))
			q++;
		sbuf_mem(sb, q, end - q);
		{
			long nl = sbuf_len(sb);
			char *nm = sbuf_done(sb);
			free(msg);
			msg = nm;
			len = nl;
		}
	}
	if (fwd) {
		struct sbuf *sb = sbuf_make();
		struct sbuf *sbo = sbuf_make();
		char *subj = msg_get(msg, len, "Subject:");
		time_t now;
		char buf[0x80];
		put_from_(sb);
		time(&now);
		strftime(buf, sizeof(buf), "%a, %d %b %Y %H:%M:%S %z", localtime(&now));
		sbuf_printf(sb, "Date: %s\n", buf);
		sbuf_printf(sb, "From: \n");
		sbuf_printf(sb, "To: \n");
		{
			char *v = strchr(subj, ':') + 1;
			while (isspace(*v))
				v++;
			sbuf_str(sb, "Subject: ");
			sbuf_str(sb, "Fwd: ");
			sbuf_mem(sb, v, pg_vlen(v));
			sbuf_str(sb, "\n");
		}
		put_id(sb);
		sbuf_printf(sb, "User-Agent: Neatmail/1.1 (https://github.com/aligrudi/neatmail)\n");
		sbuf_str(sbo, "\n-------- Original Message --------\n");
		sbuf_mem(sbo, msg, len);
		put_body(sb, sbuf_buf(sbo));
		sbuf_free(sbo);
		{
			long nl = sbuf_len(sb);
			char *nm = sbuf_done(sb);
			free(msg);
			msg = nm;
			len = nl;
		}
	}
	xwrite(1, msg, len);
	free(msg);
	return 0;

usage:
	puts(NEATMAIL_USAGE);
	return 1;
}
