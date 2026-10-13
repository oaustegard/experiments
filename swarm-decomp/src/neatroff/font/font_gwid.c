int font_gwid(void *gf, void *ff, int n, int s)
{
	struct font *g = gf;
	struct font *f = ff;
	int w, a;

	if (f == NULL) {
		if (g->cs[0] == 0) {
			w = font_wid(gf, n, s);
			goto bd;
		}
		f = g;
	}
	if (f->cs[0] != 0) {
		a = f->cs[1] ? f->cs[1] : n;
		if (g->zoom)
			a = (a * g->zoom + 500) / 1000;
		a = a * dev_res / 72 * f->cs[0] / 36;
		return a;
	}
	w = font_wid(gf, n, s);
	if (g->track[0] < 0) {
		if (g->track[2] < 0) {
			if (n > g->track[0] && n < g->track[2])
				w += ((long)((g->track[2] - n) * g->track[1] + (n - g->track[0]) * g->track[3]) * (dev_res / 72)) / (g->track[2] - g->track[0]);
		} else if (n >= g->track[2]) {
			w += (dev_res / 72) * g->track[3];
		} else if (n > g->track[0]) {
			w += ((long)((g->track[2] - n) * g->track[1] + (n - g->track[0]) * g->track[3]) * (dev_res / 72)) / (g->track[2] - g->track[0]);
		}
	} else if (n <= g->track[0]) {
		w += (dev_res / 72) * g->track[1];
	} else if (g->track[2] >= 0) {
		if (n >= g->track[2])
			w += (dev_res / 72) * g->track[3];
		else
			w += ((long)((g->track[2] - n) * g->track[1] + (n - g->track[0]) * g->track[3]) * (dev_res / 72)) / (g->track[2] - g->track[0]);
	}
bd:
	return f->bd ? w + f->bd - 1 : w;
}
