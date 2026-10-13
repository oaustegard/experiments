
static void valuerecord_print(int mask, const unsigned char *p)
{
	int v[8] = {0};
	int off = 0;
	int i;

	for (i = 0; i < 8; i++) {
		if (mask & (1 << i)) {
			unsigned short w = *(const unsigned short *)(p + off);
			v[i] = uwid((short)((w << 8) | (w >> 8)));
			off += 2;
		}
	}
	if (mask)
		printf(":%+d%+d%+d%+d", v[0], v[1], v[2], v[3]);
}
