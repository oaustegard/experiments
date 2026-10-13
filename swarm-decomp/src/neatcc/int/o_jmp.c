void o_jmp(long x)
{
	ic_put(0x1000, 0, 0, x);
}
