static int ren_traps(int a, int b, int c)
{
	int t0 = trap_pos(a);
	int t = trap_pos(a);
	int r;

	if (t < 0)
		return 0;
	if (!cdiv) {
		if (t - *nreg('p') >= 0)
			return 0;
	}
	if (b - t < 0)
		return 0;
	if (a < t0 && (c & 1))
		ren_sp(t0 - a, 0);
	r = trap_first(a);
	if (cdiv) {
		int m = cdiv->f18;
		if (!m || a - cdiv->f14 >= 0)
			m = -1;
		trap_exec(m);
	} else {
		trap_exec(r < 0 ? -1 : treg[r]);
	}
	return 1;
}
