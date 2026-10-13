int
src_macro(char *name)
{
	int i, n;

	n = nmacros;
	for (i = 0; i < n; i++)
		if (strcmp(macros[i].name, name) == 0)
			return 1;
	if (i == n)
		return 0;
	return 1;
}
