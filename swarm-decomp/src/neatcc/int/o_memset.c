void o_memset(void)
{
	long n = iv_n - 3;

	iv_n = n;
	ic_put(0x800, (int)iv[n], (int)iv[n + 1], (int)iv[n + 2]);
}
