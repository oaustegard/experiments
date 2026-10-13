int font_swid(void *f, int n, int s)
{
	struct font *F = f;
	int w, a;

	if (F->cs[0] != 0) {
		a = F->cs[1] ? F->cs[1] : n;
		if (F->zoom)
			a = (a * F->zoom + 500) / 1000;
		a = a * dev_res / 72 * F->cs[0] / 36;
		return a;
	}
	w = font_wid(f, n, (s * F->spacewidth + 6) / 12);
	return F->bd ? w + F->bd - 1 : w;
}
