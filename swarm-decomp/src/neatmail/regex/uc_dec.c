static int uc_dec(const char *s)
{
	int c0 = (unsigned char) s[0];
	unsigned char c1, c2, c3;

	if ((c0 & 0xc0) != 0xc0)
		return c0;
	c1 = s[1];
	if (!c1)
		return c0;
	if (!(c0 & 0x20))
		return ((c0 << 6) & 0x7c0) | (c1 & 0x3f);
	c2 = s[2];
	if (!c2)
		return ((c0 << 6) & 0x7c0) | (c1 & 0x3f);
	if (!(c0 & 0x10) || !s[3])
		return ((c0 << 12) & 0xffff) + (((c1 << 6) & 0xfc0) + (c2 & 0x3f));
	c3 = s[3];
	if (!(c0 & 8))
		return ((c1 << 12) & 0x3f000) | ((c2 << 6) & 0xfc0) | (c3 & 0x3f) | ((c0 << 18) & 0x1c0000);
	return c0;
}
