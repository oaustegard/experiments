static void tok_len(const char *s, int n, int a, int c, int b)
{
	printf(".nr %s 0\\w'%s'\n", nregname(n), s);
	if (a)
		printf(".nr %s 0\\n[bblly]-\\n[bbury]-2\n", nregname(a));
	if (b)
		printf(".nr %s 0\\n[bblly]-1\n", nregname(b));
	if (c)
		printf(".nr %s 0-\\n[bbury]-1\n", nregname(c));
}
