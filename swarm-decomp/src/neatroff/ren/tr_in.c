void tr_in(struct tr_args *a)
{
	int n;
	int *p;

	if (a->arg[0]) {
		int *q = nreg('i');
		char *s = a->arg[0];
		int v = *q;

		n = eval_re(s, v, 'm');
	} else {
		n = *nreg(map(".i0"));
	}
	if (*(char *)a->n == c_cc)
		ren_br();
	p = nreg('i');
	*nreg(map(".i0")) = *p;
	*nreg('i') = n < 0 ? 0 : n;
	*nreg(map(".ti")) = -1;
}
