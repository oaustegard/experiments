static int zwid(void)
{
	struct glyph *g;
	int wid;
	int size;

	g = dev_glyph("0", *nreg('f'));
	if (g == 0)
		return 0;
	wid = g->a48;
	size = *nreg('s');
	return font_gwid(g->font, dev_font(*nreg('f')), size, wid);
}
