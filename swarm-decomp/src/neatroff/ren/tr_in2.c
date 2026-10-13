void tr_in2(struct tr_args *a)
{
	int n;
	int *p;

	if (a->arg[0]) {
		int *q = nreg('I');
		char *s = a->arg[0];
		int v = *q;

		n = eval_re(s, v, 'm');
	} else {
		n = *nreg(map(".I0"));
	}
	if (*(char *)a->n == c_cc)
		ren_br();
	p = nreg('I');
	*nreg(map(".I0")) = *p;
	*nreg('I') = n < 0 ? 0 : n;
	*nreg(map(".tI")) = -1;
}
