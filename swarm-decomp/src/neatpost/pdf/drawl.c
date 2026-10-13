void drawl(int h, int v)
{
	o_h += h;
	o_v += v;
	sbuf_printf(pg, "%s l\n", pdfpos00(o_h * 100, o_v * 100));
}
