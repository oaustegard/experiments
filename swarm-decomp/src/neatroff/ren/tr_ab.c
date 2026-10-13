void tr_ab(struct tr_args *a)
{
	fprintf(stderr, "%s\n", a->arg[0]);
	ren_aborted = 1;
}
