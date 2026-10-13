static int bt_op(int a, int b)
{
	unsigned x = a & 0xf;
	unsigned y = b & 0xf;
	unsigned m = x >= y ? x : y;
	int hi = b & 0x10;

	if (a & 0x10) {
		if (y < x || hi)
			return ((a | b) & 0x10) | (m < 4 ? 4 : m);
	} else {
		if (x < y || !hi)
			return ((a | b) & 0x10) | (m < 4 ? 4 : m);
	}
	return (int)m >= 4 ? (int)m : 4;
}
