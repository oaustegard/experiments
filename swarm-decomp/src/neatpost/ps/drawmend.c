#include "ps_globals.h"
#include "outf.h"

void drawmend(const char *s)
{
	draw_path = 0;
	draw_point = 0;
	outf("%s grestore\n", s);
}
