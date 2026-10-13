#include "wb.h"
#include "libc.h"
#include "fn_wb_init.h"
#include "fn_wb_buf.h"
#include "fn_wb_done.h"
#include "wb_hmov.h"
#include "wb_put.h"
#include "fn_sbuf_append.h"
#include "fn_map.h"
#include "fn_nreg.h"

static void ren_lnum(struct sbuf *sb)
{
	struct wb w;
	char buf[2][0x10] = {{0}, {0}};
	char *s = buf[0];
	char *p;
	char c;
	int n;
	size_t len;

	wb_init(&w);
	if (*nreg(map(".nn")) <= 0) {
		n = *nreg(map("ln"));
		if (n % *nreg(map(".nM")) == 0)
			sprintf(s, "%d", *nreg(map("ln")));
	}
	wb_hmov(&w, *nreg(map(".nI")) * zwid());
	len = strlen(s);
	if (len <= 2)
		wb_hmov(&w, zwid() * (3 - len));
	c = s[0];
	if (c != 0) {
		p = s + 1;
		do {
			buf[1][0] = c;
			wb_put(&w, buf[1]);
			c = *p++;
		} while (c != 0);
	}
	wb_hmov(&w, *nreg(map(".nS")) * zwid());
	sbuf_append(sb, wb_buf(&w));
	wb_done(&w);
	if (*nreg(map(".nn")) > 0)
		--*nreg(map(".nn"));
	else
		++*nreg(map("ln"));
}
