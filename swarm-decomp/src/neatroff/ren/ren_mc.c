#include "wb.h"
#include "env_mc.h"
#include "fn_wb_init.h"
#include "wb_putexpand.h"
#include "fn_wb_buf.h"
#include "fn_wb_done.h"
#include "fn_sbuf_append.h"
#include "fn_map.h"
#include "fn_nreg.h"
#include "wb_hmov.h"

static void ren_mc(struct sbuf *sb, int a, int b)
{
	struct wb w;
	int l;
	int n;

	wb_init(&w);
	n = a + b;
	l = *nreg('l');
	l += *nreg(map(".mcn"));
	if (n < l) {
		int x;

		x = *nreg('l');
		wb_hmov(&w, *nreg(map(".mcn")) + x - a - b);
	}
	wb_putexpand(&w, env_mc());
	sbuf_append(sb, wb_buf(&w));
	wb_done(&w);
}
