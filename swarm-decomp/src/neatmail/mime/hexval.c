static int hexval(int c)
{
	if ((unsigned)(c - '0') <= 9)
		return c - '0';
	if ((unsigned)(c - 'A') <= 5)
		return c - 'A' + 10;
	if ((unsigned)(c - 'a') <= 5)
		return c - 'a' + 10;
	return 0;
}
