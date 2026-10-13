void tr_mk(struct tr_args *a)
{
	if (a->arg[0]) {
		int v = *nreg('d');
		num_set(map(a->arg[0]), v);
	} else {
		int *p = nreg('d');
		*nreg(map(".mk")) = *p;
	}
}
