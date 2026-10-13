static int uc_len(const char *s)
{
	int c = (unsigned char) s[0];

	if ((c & 0xc0) != 0xc0 || s[1] == 0)
		return c > 0;
	if (!(c & 0x20) || s[2] == 0)
		return 2;
	if (!(c & 0x10) || s[3] == 0)
		return 3;
	return (c & 8) ? 1 : 4;
}
