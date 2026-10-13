#include <stdlib.h>
#include <string.h>
#include "fn_sbuf.h"
#include "fn_sbuf_str.h"
#include "fn_base64.h"
#include "fileread.h"

void put_body(struct sbuf *sb, const char *body)
{
	int i;

	sbuf_printf(sb, "MIME-Version: 1.0\n");
	if (parts_n) {
		sbuf_printf(sb, "Content-Type: multipart/mixed; boundary=%s\n", "neatmail-boundary");
		sbuf_printf(sb, "\nMulti-part MIME message.\n");
		sbuf_printf(sb, "--%s\n", "neatmail-boundary");
		sbuf_printf(sb, "Content-Type: text/plain; charset=utf-8\n");
		sbuf_printf(sb, "Content-Transfer-Encoding: 8bit\n");
		sbuf_printf(sb, "\n");
		sbuf_str(sb, body);
		for (i = 0; i < parts_n; i++) {
			int len = 0;
			char *p, *base, *data, *enc;

			sbuf_printf(sb, "--%s\n", "neatmail-boundary");
			sbuf_printf(sb, "Content-Type: application/octet-stream\n");
			p = parts[i];
			base = strrchr(p, '/');
			if (base)
				p = base + 1;
			sbuf_printf(sb, "Content-Disposition: attachment; filename=\"%s\"\n", p);
			sbuf_printf(sb, "Content-Transfer-Encoding: base64\n");
			sbuf_printf(sb, "\n");
			data = fileread(parts[i], &len);
			if (data) {
				enc = base64((const unsigned char *)data, len);
				sbuf_mem(sb, enc, strlen(enc));
				free(enc);
			}
			free(data);
		}
		sbuf_printf(sb, "--%s--\n", "neatmail-boundary");
		return;
	}
	sbuf_printf(sb, "Content-Type: text/plain; charset=utf-8\n");
	sbuf_printf(sb, "Content-Transfer-Encoding: 8bit\n");
	sbuf_printf(sb, "\n");
	sbuf_str(sb, body);
}
