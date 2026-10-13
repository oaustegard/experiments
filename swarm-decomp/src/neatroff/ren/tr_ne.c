void tr_ne(struct tr_args *a)
{
	int n;
	int d;

	if (a->arg[0])
		n = eval(a->arg[0], 'v');
	else
		n = *nreg('v');
	if (bp_first && !cdiv) {
		bp_first = 0;
		ren_page(1);
		return;
	}
	d = *nreg('d');
	if (!ren_traps(*nreg('d'), d + n - 1, 1))
		ren_pagelimit(n - 1);
}
