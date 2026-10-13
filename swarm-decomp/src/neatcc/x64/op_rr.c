static void op_rr(int op, int r, int m, int sz)
{
	int rm = m & 7;
	int reg = (r << 3) & 0x38;

	op_x(op, r, m, sz);
	op_put((char) (reg | rm) | 0xc0, 1);
}
