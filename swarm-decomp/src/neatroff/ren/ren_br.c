static int ren_br(void)
{
	struct wb *w;
	void *f;
	int r;

	if (bp_first && !cdiv) {
		bp_first = 0;
		ren_page(1);
	}
	w = env_wb();
	for (;;) {
		f = env_fmt();
		if (!fmt_word(f, w))
			break;
		f = env_fmt();
		while (fmt_morelines(f))
			ren_passline(f);
	}
	wb_reset(w);
	for (;;) {
		f = env_fmt();
		r = fmt_fill(f, 1);
		if (!r)
			break;
		f = env_fmt();
		while (fmt_morelines(f))
			ren_passline(f);
	}
	f = env_fmt();
	r = 0;
	while (fmt_morelines(f))
		r = ren_passline(f);
	return r;
}
