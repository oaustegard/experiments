void o_tmpdrop(int n)
{
	long k;

	if (n < 0)
		n = iv_n;
	k = iv_n - n;
	iv_n = k < 0 ? 0 : k;
}
