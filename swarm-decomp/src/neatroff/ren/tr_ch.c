void tr_ch(struct tr_args *a)
{
	int m;
	int i;

	if (!a->arg[0])
		return;
	m = map(a->arg[0]);
	for (i = 0; i < ntraps; i++) {
		int t = treg[i];
		if (m == t) {
			if (a->arg[1])
				tpos[i] = eval(a->arg[1], 'v');
			else
				treg[i] = -1;
			return;
		}
	}
}
