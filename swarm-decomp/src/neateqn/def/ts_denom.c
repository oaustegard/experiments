int ts_denom(int s)
{
	if ((unsigned int)s <= 1)
		return 3;
	return ts_sub(s);
}
