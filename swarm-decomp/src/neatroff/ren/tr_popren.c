void tr_popren(struct tr_args *a)
{
	if (a->arg[0])
		ren_level = strtol(a->arg[0], 0, 10);
	else
		ren_level = 0;
}
