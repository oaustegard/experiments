void
src_define(char *name, char *def)
{
	int i;
	size_t len;

	for (i = 0; i < nmacros; i++)
		if (strcmp(macros[i].name, name) == 0)
			break;
	if (i == nmacros) {
		if (nmacros > 0x1ff)
			return;
		nmacros++;
	}
	strcpy(macros[i].name, name);
	free(macros[i].body);
	len = strlen(def) + 1;
	macros[i].body = memcpy(malloc(len), def, len);
}
