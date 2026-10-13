void o_code(char *name, char *s, long n)
{
	out_def(name, 1, mem_len(&cs), 0);
	mem_put(&cs, s, n);
}
