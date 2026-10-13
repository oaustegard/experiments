static void ren_sp(int n, int f)
{
	int r12, r13;

	if (bp_first && !cdiv) {
		bp_first = 0;
		ren_page(1);
	}
	if (n == 0) {
		r12 = 0;
		if (ren_div && ren_divvs && *nreg('u') == 0)
			r12 = ren_divvs;
		ren_divvs = 0;
		*nreg(map(".ns")) = 0;
		if (r12) {
			r13 = r12;
		} else {
			r13 = 0;
			r12 = *nreg('v');
		}
	} else {
		ren_divvs = 0;
		*nreg(map(".ns")) = 0;
		r13 = n;
		r12 = n;
	}
	*nreg('d') += r12;
	if (*nreg('d') > *nreg('h')) {
		int *p = nreg('d');
		*nreg('h') = *p;
	}
	if (cdiv) {
		if (!(f & 1)) {
			if (n == 0) {
				sbuf_printf(&cdiv->buf, "%c%s %du\n", c_cc, "\aV", *nreg('v'));
			} else {
				if (r13 == 0)
					r13 = *nreg('v');
				sbuf_printf(&cdiv->buf, "%csp %du\n", c_cc, r13);
			}
		}
	}
	if (cdiv)
		return;
	{
		int *p = nreg('d');
		*nreg(map("nl")) = *p;
	}
}
