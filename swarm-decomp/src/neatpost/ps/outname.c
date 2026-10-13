#include "outf.h"
#include "ps_state.h"
#include "ps_extern.h"
#include "ps_vars.h"

void outname(int n, char *names, int *pages, int *tops)
{
	int i;

	o_flush();
	for (i = 0; i < n; i++) {
		outf("[ /Dest /%s", names + i * 64);
		outf(" /Page %d", pages[i]);
		if (tops[i] > 0)
			outf(" /View [/XYZ null %d null]", (ps_height - tops[i]) * 72 / dev_res);
		outf(" /DEST pdfmark\n");
	}
}
