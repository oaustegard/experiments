#include "obj_map.h"
void docheader(char *title, int w, int h, int lw)
{
	if (title)
		outinfo("Title", title);
	obj_map();
	pdf_root = obj_map();
	pdf_pages = obj_map();
	pdfout("%%PDF-1.6\n\n");
	pdf_width = (w * 72 + 127) / 254;
	pdf_height = (h * 72 + 127) / 254;
	pdf_linewid = lw;
}
