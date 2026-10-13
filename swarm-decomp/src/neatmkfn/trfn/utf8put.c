static void utf8put(char **sp, int c)
{
	int k, b;

	if (c > 0xffff) {
		b = (c >> 18) | 0xfffffff0;
		k = 2;
	} else if (c > 0x7ff) {
		b = (c >> 12) | 0xffffffe0;
		k = 1;
	} else if (c > 0x7f) {
		b = (c >> 6) | 0xffffffc0;
		k = 0;
	} else {
		char *p = *sp;
		*sp = p + 1;
		*p = c > 0 ? c : 0x20;
		**sp = 0;
		return;
	}
	*(*sp)++ = b;
	for (; k >= 0; k--)
		*(*sp)++ = ((c >> (6 * k)) & 0x3f) | 0xffffff80;
	**sp = 0;
}
