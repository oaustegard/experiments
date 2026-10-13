static void op_rm(int op, int r, int m, int disp, int sz)
{
	int mod = ((char) disp != disp) + 1;
	int n = (char) disp != disp ? 4 : 1;
	int rm = m & 7;
	int r8 = (r << 3) & 0x38;
	int reg = r8 | rm;

	if (disp == 0 && rm != 5) {
		op_x(op, r, m, sz);
		op_put(reg, 1);
		if (rm == 4)
			op_put(0x24, 1);
		return;
	}
	op_x(op, r, m, sz);
	op_put((mod << 6) | reg, 1);
	if (rm == 4)
		op_put(0x24, 1);
	op_put(disp, n);
}
