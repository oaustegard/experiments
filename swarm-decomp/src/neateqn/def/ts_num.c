int ts_num(int s)
{
	if ((unsigned int)s <= 1)
		return 2 + (s != 0);
	return ts_sup(s);
}
