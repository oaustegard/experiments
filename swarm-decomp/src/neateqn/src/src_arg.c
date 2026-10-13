int
src_arg(int n)
{
	int have;
	char *a;

	have = esrc->nargs;
	if (have) {
		a = esrc->arg[n - 1];
		if (a)
			src_push(a, NULL);
	}
	return !have;
}
