void tr_ce(struct tr_args *a)
{
	int n;

	if (*(char *)a->n == c_cc)
		ren_br();
	n = 1;
	if (a->arg[0])
		n = strtol(a->arg[0], 0, 10);
	*nreg(map(".ce")) = n;
}
