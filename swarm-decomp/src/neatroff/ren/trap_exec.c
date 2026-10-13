#include <stdio.h>
#include "render_rec.h"
#include "ren_page.h"
#include "trap_exec.h"

static void trap_exec(int t)
{
	char buf[16];
	int u;

	if (!ren_partial)
		goto plain;
	if (ren_un)
		goto unbuf;
	if (str_get(t)) {
		sprintf(buf, "%c%s %d\n", c_cc, "\x07P", ren_level);
		in_push(buf, 0);
		in_push(str_get(t), 0);
		in_push("\n", 0);
		ren_level++;
		render_rec(ren_level);
		fmt_suppressnl(env_fmt());
	}
	return;
unbuf:
	u = ren_unbuf[0];
	if (str_get(t)) {
		sprintf(buf, "%c%s %d\n", c_cc, "\x07P", ren_level);
		in_push(buf, 0);
		in_push(str_get(t), 0);
		if (u == '\n') {
			ren_level++;
			render_rec(ren_level);
			return;
		}
		in_push("\n", 0);
		ren_level++;
		render_rec(ren_level);
		fmt_suppressnl(env_fmt());
	}
	return;
plain:
	if (str_get(t)) {
		sprintf(buf, "%c%s %d\n", c_cc, "\x07P", ren_level);
		in_push(buf, 0);
		in_push(str_get(t), 0);
		ren_level++;
		render_rec(ren_level);
	}
}
