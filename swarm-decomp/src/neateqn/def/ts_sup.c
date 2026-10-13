int ts_sup(int s)
{
	int a = s >> 4;
	if (a > 0)
		a = 1;
	return ((a + 1) << 4) | (s & 1);
}
