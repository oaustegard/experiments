static void lnput(char *s, int n)
{
	long len = n;

	if (n < 0)
		len = strlen(s);
	write(1, s, len);
}
