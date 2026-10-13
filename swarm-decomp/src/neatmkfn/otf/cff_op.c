int
cff_op(const unsigned char *p, int off, int *val)
{
	int b, v;
	int j;

	b = p[off];
	if ((unsigned)(b - 0x20) <= 0xd6) {
		*val = b - 0x8b;
		return 1;
	}
	if ((unsigned)(b - 0xf7) <= 3) {
		v = ((b - 0xf7) << 8) + p[off + 1] + 0x6c;
		*val = v;
		return 2;
	}
	if ((unsigned)(b - 0xfb) <= 3) {
		v = ((0xfb - b) << 8) - p[off + 1] - 0x6c;
		*val = v;
		return 2;
	}
	if (b == 0x1c) {
		v = (p[off + 1] << 8) | p[off + 2];
		*val = v;
		return 3;
	}
	if (b == 0x1d) {
		v = (p[off + 1] << 24) | (p[off + 2] << 16) | (p[off + 4]) | (p[off + 3] << 8);
		*val = v;
		return 5;
	}
	if (b == 0x1e) {
		j = 2;
		for (;;) {
			int nb = p[off + j - 1];
			if ((nb & 0xf) == 0xf)
				break;
			if ((nb & 0xf0) == 0xf0)
				break;
			j++;
			if (j == 0x21)
				break;
		}
		*val = 0;
		return j;
	}
	*val = b;
	return 1;
}
