static void ren_ejectpage(int n)
{
	int p;
	int t;

	if (bp_first && !cdiv) {
		bp_first = 0;
		ren_page(1);
	}
	bp_ejected = *nreg('%');
	if (n)
		ren_br();
	for (;;) {
		if (*nreg('%') != bp_ejected)
			return;
		if (cdiv)
			return;
		p = *nreg('p');
		t = trap_pos(*nreg('d'));
		if (t < 0 || (!cdiv && t - *nreg('p') >= 0) || p < t) {
			bp_ejected = 0;
			ren_page(0);
			continue;
		}
		ren_traps(*nreg('d'), *nreg('p'), 1);
	}
}
