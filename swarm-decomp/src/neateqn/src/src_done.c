void
src_done(void)
{
	int i;

	for (i = 0; i < nmacros; i++)
		free(macros[i].body);
}
