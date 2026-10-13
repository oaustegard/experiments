int utf8len(int c)
{
	int n;

	if (~c & 0xc0)
		return c > 0;
	n = 2;
	if (c & 0x20) {
		n = 3;
		if (c & 0x10)
			n = (c & 8) ? 1 : 4;
	}
	return n;
}
