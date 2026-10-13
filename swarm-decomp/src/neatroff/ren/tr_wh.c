void tr_wh(struct tr_args *a)
{
	int pos;
	int m;
	int i;
	int j;
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
		if (t != pos)
			continue;
		goto found;
	}
	if (!a->arg[1])
		return;
	m = map(a->arg[1]);
	n = ntraps;
	for (j = 0; j < n; j++)
		if (treg[j] == -1) {
			i = j;
			goto set;
		}
	i = n;
	ntraps = n + 1;
	goto set;
found:
	if (!a->arg[1]) {
		treg[i] = -1;
		return;
	}
	m = map(a->arg[1]);
set:
	tpos[i] = pos;
	treg[i] = m;
}
