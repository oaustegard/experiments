void tr_l2r(struct tr_args *a)
{
	dir_do = 1;
	if (*(char *)a->n == c_cc)
		ren_br();
	*nreg(map(".td")) = 0;
	*nreg(map(".cd")) = 0;
}
