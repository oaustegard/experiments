static void op_rr(int op, int r, int m, int sz)
{
	int mod = ((r << 3) & 0x38) | (m & 7);

	op_x(op, r, m, sz);
	mod |= 0xc0;
	op_put((char) mod, 1);
}
