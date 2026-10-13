void tr_em(struct tr_args *a)
{
	if (a->arg[0])
		trap_em = map(a->arg[0]);
	else
		trap_em = -1;
}
