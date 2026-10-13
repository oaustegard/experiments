int font_getcs(void *f)
{
	struct font *F = f;

	return F->cs[0];
}
