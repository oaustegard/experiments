void drawa(int h, int v, int dh, int dv)
{
	h += dh;
	v += dv;
	o_h += h;
	o_v += v;
	sbuf_printf(pg, "%s l\n", pdfpos00(o_h * 100, o_v * 100));
}
