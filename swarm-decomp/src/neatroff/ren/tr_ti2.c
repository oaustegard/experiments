void tr_ti2(struct tr_args *a)
{
	if (*(char *)a->n == c_cc)
		ren_br();
	if (a->arg[0]) {
		int *p = nreg('I');
		char *s = a->arg[0];
		int v = *p;
		int *q = nreg(map(".tI"));

		*q = eval_re(s, v, 'm');
	}
}
