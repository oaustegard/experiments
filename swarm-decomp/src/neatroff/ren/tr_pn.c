void tr_pn(struct tr_args *a)
{
	if (a->arg[0])
		bp_next = eval_re(a->arg[0], *nreg(map("%")), 0);
}
