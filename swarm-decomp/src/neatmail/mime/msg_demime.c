#include <ctype.h>
#include <string.h>
#include "fn_sbuf.h"
#include "fn_startswith.h"
#include "fn_hdrlen.h"
#include "fn_msg_demime.h"
#include "msg_hdrdec2.h"
#include "dec_qp.h"
#include "toutf8.h"
#include "hdr_nextfield.h"

int msg_demime(char *s, long n, char **out, long *outn)
{
	struct sbuf *ob = sbuf_make();
	struct sbuf *sb;
	char *p = s;
	char *e = s + n;
	char *s0 = s;
	const char *cs = NULL;
	char *chunk, *end, *nl, *np, *tok, *v, *q, *r, *v2, *hend;
	int nb = 0, flags = 0, hl, k;
	long l;
	char bnd[8][128];

	for (;;) {
		if (!(p == s0 || (nb && p < e)))
			break;
		sb = sbuf_make();
		cs = NULL;
		flags = 0;
		if (p < e && p != NULL) {
			while (*p != '\n') {
				nl = memchr(p, '\n', e - p);
				if (!nl)
					goto hdr_end;
				np = nl + 1;
				while (np < e && *np != '\n' && isspace(*np)) {
					nl = memchr(np, '\n', e - np);
					if (!nl)
						goto hdr_end;
					np = nl + 1;
				}
				if (startswith(p, "Content-Type:")) {
					v = strchr(p, ':') + 1;
					hl = hdrlen(p, e - p);
					hend = p + hl;
					for (tok = v; ; ) {
						while (tok < hend && isspace(*tok))
							tok++;
						if (startswith(tok, "text"))
							flags |= 1;
						if (startswith(tok, "multipart"))
							flags |= 2;
						if (startswith(tok, "charset")) {
							q = memchr(tok, '=', hend - tok);
							if (!q)
								cs = NULL;
							else if (q + 14 < hend && startswith(q, "=\"windows-1256\""))
								cs = "windows-1256";
							else if (q + 12 < hend && startswith(q, "=\"iso-8859-6\""))
								cs = "iso-8859-6";
							else
								cs = NULL;
						}
						if (startswith(tok, "boundary")) {
							q = memchr(tok, '=', hend - tok);
							if (q) {
								if (q[1] == '"') {
									v2 = q + 2;
									r = memchr(v2, '"', hend - v2);
									if (r) {
										l = r - v2;
										goto store_bnd;
									}
								} else {
									v2 = q + 1;
									if (v2 >= hend) {
										l = 0;
										goto store_bnd;
									}
									for (r = v2; r < hend && !isspace(*r) && *r != ';'; r++)
										;
									l = r - v2;
									goto store_bnd;
								}
							}
						}
						goto next_tok;
store_bnd:
						memcpy(bnd[nb], "--", 2);
						memcpy(bnd[nb] + 2, v2, l);
						bnd[nb][l + 2] = 0;
						nb++;
next_tok:
						tok = hdr_nextfield(tok, hend);
						if (!tok)
							break;
					}
				}
				if (startswith(p, "Content-Transfer-Encoding:")) {
					v = strchr(p, ':') + 1;
					hl = hdrlen(p, e - p);
					hend = p + hl;
					for (tok = v; ; ) {
						while (tok < hend && isspace(*tok))
							tok++;
						if (startswith(tok, "quoted-printable"))
							flags |= 0x20;
						if (startswith(tok, "base64"))
							flags |= 0x40;
						tok = hdr_nextfield(tok, hend);
						if (!tok)
							break;
					}
				}
				msg_hdrdec2(sb, p, np);
				if (np >= e) {
					p = np;
					break;
				}
				p = np;
			}
		}
hdr_end:
		p++;
		sbuf_chr(sb, '\n');
		if (nb) {
			chunk = p;
			end = e;
			while (p < e) {
				if (startswith(p, bnd[nb - 1])) {
					end = p;
					break;
				}
				nl = memchr(p, '\n', e - p);
				if (!nl)
					break;
				p = nl + 1;
				if (p >= e)
					break;
			}
			p = chunk;
		} else {
			end = e;
		}
		if ((flags & 1) && (flags & 0x20))
			dec_qp(sb, p, end, 0);
		else if ((flags & 1) && (flags & 0x40)) {
			dec_b64(sb, p, end);
			sbuf_chr(sb, '\n');
		} else
			sbuf_mem(sb, p, end - p);
		p = end;
		l = sbuf_len(sb);
		toutf8(cs, ob, sbuf_buf(sb), l);
		sbuf_free(sb);
		if (nb) {
			k = nb - 1;
			l = strlen(bnd[k]);
			if (p[l] == '-' && p[l + 1] == '-')
				nb = k;
			nl = memchr(p, '\n', e - p);
			l = nl ? nl + 1 - p : e - p;
			sbuf_mem(ob, p, l);
			p += l;
		}
	}
	sbuf_chr(ob, 0);
	*outn = sbuf_len(ob) - 1;
	*out = sbuf_done(ob);
	return 0;
}
