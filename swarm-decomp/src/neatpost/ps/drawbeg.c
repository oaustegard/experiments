#include "ps_state.h"
#include "ps_vars.h"
#include "ps_globals.h"
#include "outf.h"

static void o_flush(void);
static void out_fontup(int f);

void drawbeg(void)
{
	o_flush();
	out_fontup(o_f);
	if (!draw_path)
		outf("newpath ");
}
