void tr_bp(struct tr_args *a)
{
	if (cdiv)
		return;
	if (a->arg[0] == 0) {
		if (*nreg(map(".ns")))
			return;
	}
	if (a->arg[0] != 0)
		bp_next = eval_re(a->arg[0], *nreg(map("%")), 0);
	ren_ejectpage(*(char *)a->n == c_cc);
}
