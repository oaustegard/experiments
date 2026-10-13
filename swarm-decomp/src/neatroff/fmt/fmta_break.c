/* line-break list */
static int fmta_break(int *w, int n, int *brk, int m)
{
	int k, x;

	if (n > 0) {
		x = w[n];
		if (x > 0) {
			k = fmta_break(w, x, brk, m);
			if (m > k) {
				brk += k;
				*brk = n - x;
			}
			return k + 1;
		}
		if (m)
			*brk = n - x;
		return 1;
	}
	if (m)
		*brk = n;
	return 0;
}
