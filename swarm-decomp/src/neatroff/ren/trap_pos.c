static int trap_pos(int d)
{
	int i = trap_first(d);

	if (bp_final > 2)
		return -1;
	if (cdiv) {
		int r = -1;

		if (cdiv->f18) {
			if (cdiv->f14 > d)
				r = cdiv->f14;
		}
		return r;
	}
	if (i < 0)
		return -1;
	if (tpos[i] < 0)
		return *nreg('p') + tpos[i];
	return tpos[i];
}
