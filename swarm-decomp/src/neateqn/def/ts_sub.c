int ts_sub(int s)
{
	s >>= 4;
	if (s > 0)
		s = 1;
	return ((s + 1) << 4) | 1;
}
