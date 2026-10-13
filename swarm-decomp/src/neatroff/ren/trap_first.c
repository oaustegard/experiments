static int trap_first(int d)
{
	int best = -1;
	int i;

	for (i = 0; i < ntraps; i++) {
		int pos;

		if (treg[i] < 0)
			continue;
		pos = trap_abs(i);
		if (pos <= d)
			continue;
		if (best < 0) {
			best = i;
			continue;
		}
		if (trap_abs(i) < trap_abs(best))
			best = i;
	}
	return best;
}
