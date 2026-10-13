static int valuerecord_small(int mask, const unsigned char *p)
{
	int kmin = mkfn_kmin;
	int off = 0;
	int i = 0;

	if (kmin <= 0)
		kmin = 1;
	do {
		if (mask & (1 << i)) {
			unsigned short w = *(const unsigned short *)(p + off);
			int v = uwid((short)((w << 8) | (w >> 8)));
			int a = v < 0 ? -v : v;

			if (a >= kmin)
				return 0;
			off += 2;
		}
		i++;
	} while (i != 8);
	return 1;
}
