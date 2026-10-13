void tr_rt(struct tr_args *a)
{
	int n;

	if (a->arg[0])
		n = eval_re(a->arg[0], *nreg('d'), 'v');
	else
		n = *nreg(map(".mk"));
	if (n >= 0) {
		if (*nreg('d') > n)
			ren_sp(n - *nreg('d'), 0);
	}
}
