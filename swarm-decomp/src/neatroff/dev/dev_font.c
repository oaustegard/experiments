void *dev_font(int n)
{
	void *p = 0;

	if ((unsigned)n <= 31)
		p = fn_font[n];
	return p;
}
