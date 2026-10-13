void tr_br(struct tr_args *a)
{
	void *f;

	if (*(char *)a->n == c_cc) {
		ren_br();
		return;
	}
	f = env_fmt();
	for (;;) {
		if (!fmt_fill(f, 0))
			break;
		while (fmt_morelines(f))
			ren_passline(f);
	}
	while (fmt_morelines(f))
		ren_passline(f);
}
