void tr_ll(struct tr_args *a)
{
	int n;
	int *p;

	if (a->arg[0]) {
		int *q = nreg('l');
		char *s = a->arg[0];
		int v = *q;

		n = eval_re(s, v, 'm');
	} else {
		n = *nreg(map(".l0"));
	}
	p = nreg('l');
	*nreg(map(".l0")) = *p;
	*nreg('l') = n < 0 ? 0 : n;
}
