int
src_expand(char *name, char **args)
{
	int i, r;

	for (i = 0; i < nmacros; i++) {
		r = strcmp(macros[i].name, name);
		if (r == 0) {
			src_push(macros[i].body, args);
			return r;
		}
	}
	return 1;
}
