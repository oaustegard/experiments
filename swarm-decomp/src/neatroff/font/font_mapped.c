int font_mapped(void *f, char *s)
{
	struct font *F = f;

	return dict_get(F->d88, s) != -1;
}
