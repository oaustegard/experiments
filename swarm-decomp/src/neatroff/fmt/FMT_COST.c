static long FMT_COST(int a, int b, int c, int d)
{
	int i;
	long x, s, u;

	x = (a - b) * 100L;
	if (c)
		x /= c;
	s = x < 0 ? -x : x;
	if (s > 4000) {
		s -= 4000;
		x = 0;
		for (i = 0; i != 14; i++)
			x += ((s >> 2 * i) & 3) << (i + 3);
		if (x > 0x2000)
			x = 0x2000;
		x += 4000;
	}
	u = x * x / 100;
	if (d)
		u *= d;
	return u;
}
