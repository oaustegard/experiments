void tr_wh(struct tr_args *a)
{
	int pos;
	int m;
	int i;
	int n;

	if (!a->arg[0])
		return;
	pos = eval(a->arg[0], 'v');
	for (i = 0; i < ntraps; i++) {
		int t;

		if (treg[i] < 0)
			continue;
		t = tpos[i];
		if (t < 0)
			t = *nreg('p') + tpos[i];
		if (t == pos) {
			if (!a->arg[1]) {
				treg[i] = -1;
				return;
			}
			m = map(a->arg[1]);
			goto set;
		}
	}
	if (!a->arg[1])
		return;
	m = map(a->arg[1]);
	n = ntraps;
	for (i = 0; i < n; i++)
		if (treg[i] == -1)
			break;
	if (i == n)
		ntraps = n + 1;
set:
	tpos[i] = pos;
	treg[i] = m;
}
