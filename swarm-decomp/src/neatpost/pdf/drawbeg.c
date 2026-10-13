void drawbeg(void)
{
	if (o_queued == 1)
		sbuf_printf(pg, ">] TJ\n");
	o_queued = 0;
	out_fontup();
	sbuf_printf(pg, "%s m\n", pdfpos00(o_h * 100, o_v * 100));
}
