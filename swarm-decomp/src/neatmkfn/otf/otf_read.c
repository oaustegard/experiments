int otf_read(void)
{
	unsigned char *b = otf_input(0);
	unsigned int tag = *(unsigned int *)b;
	int i, n;

	if (tag == 0x100 || tag == 0x4f54544f)
		return otf_offsettable(b, b);
	if (tag != 0x66637474)
		return 1;
	n = __builtin_bswap32(*(unsigned int *)(b + 8));
	for (i = 0; i < n; i++)
		otf_offsettable(b, b + __builtin_bswap32(*(unsigned int *)(b + 12 + 4 * i)));
	return 0;
}
