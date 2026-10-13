static void tok_dim(const char *s, int n, int a, int b)
{
	printf(".nr %s 0\\w'%s'\n", nregname(n), s);
	if (a)
		printf(".nr %s 0-\\n[bbury]\n", nregname(a));
	if (b)
		printf(".nr %s 0\\n[bblly]\n", nregname(b));
}
