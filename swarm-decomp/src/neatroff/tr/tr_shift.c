static void tr_shift(struct tr_args *a)
{
	char *s = a->arg[0];
	int n, k;

	if (!s) {
		in_shift();
		return;
	}
	n = (int)strtol(s, 0, 10);
	if (n > 0) {
		k = n - 1;
		if (!(n & 1))
			goto even;
		in_shift();
		k = n - 2;
		if (n == 1)
			return;
	even:
		do {
			in_shift();
			k -= 2;
			in_shift();
		} while (k != -1);
	}
}
