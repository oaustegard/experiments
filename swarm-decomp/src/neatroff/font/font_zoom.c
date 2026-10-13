int font_zoom(void *f, int n)
{
	struct font *F = f;
	int z = F->zoom;

	if (z)
		n = (z * n + 500) / 1000;
	return n;
}
