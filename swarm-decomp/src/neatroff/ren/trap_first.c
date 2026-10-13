static int trap_first(int d)
{
	int best = -1;
	int i;

	for (i = 0; i < ntraps; i++) {
		int pos;

		if (treg[i] < 0)
			continue;
		pos = tpos[i];
		if (pos < 0)
			pos = *nreg('p') + tpos[i];
		if (pos > d) {
			if (best < 0) {
				best = i;
				continue;
			}
			if (trap_abs(i) - trap_abs(best) >= 0)
				continue;
			best = i;
		}
	}
	return best;
}
