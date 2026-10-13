#include <ctype.h>
#include "outf.h"
#include "ps_state.h"
#include "ps_extern.h"
#include "ps_vars.h"

void outlink(char *dest, int x, int y)
{
	char *prefix = "/Page ";
	int c;

	o_flush();
	c = (unsigned char)*dest;
	if (c == '#') {
		dest++;
		prefix = "/Dest /";
	} else if (!isdigit(c)) {
		outf("[ /Rect [ %d %d t %d %d t ] /Action << /Subtype /URI /URI %s >> /Open true /Subtype /Link /LNK pdfmark\n", o_h, o_v, o_h + x, o_v + y, pdftext_static(dest));
		return;
	}
	outf("[ /Rect [ %d %d t %d %d t ] %s%s /Subtype /Link /LNK pdfmark\n", o_h, o_v, o_h + x, o_v + y, prefix, dest);
}
