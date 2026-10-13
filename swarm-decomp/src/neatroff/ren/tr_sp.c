void tr_sp(struct tr_args *a)
{
	int n;
	int br = 0;

	if (*(char *)a->n == c_cc)
		br = ren_br();
	if (a->arg[0])
		n = eval(a->arg[0], 'v');
	else
		n = *nreg('v');
	if (n == 0)
		return;
	if (*nreg(map(".ns"))) {
		if (!ren_div)
			return;
	}
	if (br)
		return;
	down(n);
}
