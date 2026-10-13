static void op_put(long v, long n)
{
	static char buf[16];
	long i = 0;

	do {
		buf[i++] = v;
		v >>= 8;
	} while (i != n);
	mem_put(&cs, buf, n);
}
