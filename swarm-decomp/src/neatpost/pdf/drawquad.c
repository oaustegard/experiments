static void drawquad(int h, int v)
{
	long x1 = (long)(o_h * 1000) + (long)(h * 500);
	long y1 = (long)(o_v * 1000) + (long)(v * 500);
	long a = (long)v * 551915 / 2000;
	long b = (long)h * 551915 / -2000;
	long p1x, p1y, p2x, p2y;
	if (!((h > 0 && v < 0) || (h < 0 && v > 0))) {
		p1x = (long)(o_h * 1000);
		p1y = (long)(o_v * 1000) + a;
		p2x = x1 + b;
		p2y = y1;
	} else {
		p1x = (long)(o_h * 1000) - b;
		p1y = (long)(o_v * 1000);
		p2x = x1;
		p2y = y1 - a;
	}
	sbuf_printf(pg, "%s ", pdfpos00(p1x / 10, p1y / 10));
	sbuf_printf(pg, "%s ", pdfpos00(p2x / 10, p2y / 10));
	sbuf_printf(pg, "%s c\n", pdfpos00(x1 / 10, y1 / 10));
	o_h += h / 2;
	o_v += v / 2;
}
