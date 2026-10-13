#include "ps_state.h"
#include "ps_extern.h"

void doctrailer(int pages)
{
	out("[");
	if (ps_title[0])
		out(" /Title %s", pdftext_static(ps_title));
	if (ps_author[0])
		out(" /Author %s", pdftext_static(ps_author));
	out(" /Creator (Neatroff) /DOCINFO pdfmark\n");
	out("%%%%Trailer\n");
	out("done\n");
	out("%%%%DocumentFonts: %s\n", o_fonts);
	out("%%%%Pages: %d\n", pages);
	out("%%%%EOF\n");
}
