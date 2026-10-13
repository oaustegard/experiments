int i_imm(int n, long v)
{
	long m = (1 << (n - 1)) - 1;

	if (m < v)
		return 0;
	return v + 1 >= -m;
}
