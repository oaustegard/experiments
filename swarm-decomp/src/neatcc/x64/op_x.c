static void op_x(int op, int r, int m, int sz)
{
	int x;

	sz &= 0xf;
	if (sz == 8) {
		if (r & 8)
			x = (m & 8) ? 0x4d : 0x4c;
		else
			x = (m & 8) ? 0x49 : 0x48;
		op_put(x, 1);
	} else {
		x = (sz == 1) << 6;
		if (r & 8) {
			x |= 4;
			if (!(m & 8)) {
				if (sz == 2)
					op_put(0x66, 1);
				x = 0x44;
				op_put(x, 1);
				goto tail;
			}
		} else if (!(m & 8)) {
			if (sz == 2)
				op_put(0x66, 1);
			else if (sz == 1)
				op_put(0x40, 1);
			goto tail;
		}
		if (sz == 2)
			op_put(0x66, 1);
		x |= 0x41;
		op_put(x, 1);
	}
tail:
	if (op & 0x10000)
		op_put((op >> 8) & 0xff, 1);
	if (sz == 1)
		op &= 0xfe;
	else
		op &= 0xff;
	op_put(op, 1);
}
