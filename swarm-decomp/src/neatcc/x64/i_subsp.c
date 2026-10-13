static void i_subsp(long n)
{
	if (n == 0)
		return;
	if ((unsigned long) (n + 0x80) <= 0xff) {
		mem_put(&cs, "H\x83\xec", 3);
		op_put(n, 1);
	} else {
		mem_put(&cs, "H\x81\xec", 3);
		op_put(n, 4);
	}
}
