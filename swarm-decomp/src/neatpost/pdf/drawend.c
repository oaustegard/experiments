void drawend(int close, int flags)
{
	int fill = 2;
	int stroke = 0;
	char *fmt;
	int w;

	if (flags) {
		fill = flags & 2;
		stroke = flags & 1;
	}
	if (l_page != page_n || l_size != o_s || l_wid != pdf_linewid || l_cap != pdf_linecap || l_join != pdf_linejoin) {
		w = o_s * pdf_linewid;
		sbuf_printf(pg, "%d.%03d w\n", w / 1000, w % 1000);
		sbuf_printf(pg, "%d J %d j\n", pdf_linecap, pdf_linejoin);
		l_page = page_n;
		l_size = o_s;
		l_wid = pdf_linewid;
		l_cap = pdf_linecap;
		l_join = pdf_linejoin;
	}
	if (fill)
		sbuf_printf(pg, "%s RG\n", pdfcolor(o_m));
	fmt = stroke ? (fill ? "b\n" : "f\n") : (close ? "s\n" : "S\n");
	sbuf_printf(pg, fmt);
	p_v = 0;
	p_h = 0;
}
