int render(void)
{
	*nreg(map("nl")) = -1;
	while (!tr_nextreq())
		;
	if (bp_first && !cdiv) {
		bp_first = 0;
		ren_page(1);
	}
	render_rec(0);
	bp_final = 3;
	if (fmt_morewords(env_fmt()))
		ren_page(1);
	ren_br();
	return ren_aborted;
}
