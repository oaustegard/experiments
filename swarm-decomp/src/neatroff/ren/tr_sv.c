void tr_sv(struct tr_args *a)
{
	int n;

	if (a->arg[0])
		n = eval(a->arg[0], 'v');
	else
		n = *nreg('v');
	*nreg(map(".sv")) = 0;
	if (f_nexttrap() >= n)
		down(n);
	else
		*nreg(map(".sv")) = n;
}
