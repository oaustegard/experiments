int
src_macro(char *name)
{
	int i;

	for (i = 0; i < nmacros; i++)
		if (strcmp(macros[i].name, name) == 0)
			return 1;
	if (i == nmacros)
		return 0;
	return 1;
}
