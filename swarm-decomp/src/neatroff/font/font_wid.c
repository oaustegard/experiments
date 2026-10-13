int font_wid(void *f, int n, int k)
{
	int u, h;

	n = font_zoom(f, n);
	if (k < 0) {
		u = dev_uwid;
		h = u / 2;
		return -((h + (-k) * n) / u);
	}
	u = dev_uwid;
	h = u / 2;
	return (h + k * n) / u;
}
