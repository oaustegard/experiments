#include <string.h>
#include "fmt.h"
#include "wb.h"
#include "fn_free.h"
#include "fn_xmalloc.h"
#include "fn_wb_init.h"
#include "fn_wb_done.h"
#include "fn_wb_buf.h"
#include "fn_wb_wid.h"
#include "fn_wb_keshideh.h"

static void fmt_keshideh(struct fmt *f, int n, int a3)
{
	struct wb w;
	struct fmtword *p;
	int i, j, m, r, count, pos;

	do {
		count = 0;
		i = n - 1;
		while (i != n - 3) {
			if (i < 0) {
				i--;
				continue;
			}
			m = (unsigned int)i / 2 + 1;
			j = i;
			do {
				p = &f->words[j];
				wb_init(&w);
				r = wb_keshideh(p->s, &w, a3);
				if (r > 0) {
					free(p->s);
					count++;
					p->s = xmalloc(strlen(wb_buf(&w)) + 1);
					strcpy(p->s, wb_buf(&w));
					p->wid = wb_wid(&w);
					a3 -= r;
				}
				wb_done(&w);
				j -= 2;
			} while (--m > 0);
			i--;
		}
	} while (count != 0);
	if (n > 0) {
		for (i = 0; i < n; i++) {
			p = &f->words[i];
			pos = (i ? p[-1].pos : 0) + p->wid + p->f14;
			p->pos = pos;
		}
	}
}
