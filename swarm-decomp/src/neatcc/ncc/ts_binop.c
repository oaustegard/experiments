static void ts_binop(int op)
{
	int kl, kr, fl, fr;

	ts_de(1);
	nts--;
	kl = ts[nts].kind;
	fl = ts[nts].ptr;
	o_tmpswap();
	ts_de(1);
	nts--;
	fr = ts[nts].ptr;
	kr = ts[nts].kind;
	o_tmpswap();
	if (fl)
		kl = 8;
	if (fr)
		kr = 8;
	kl = bt_op(kl, kr);
	if (op == 'A' || op == 'B') {
		kl &= 0xf;
		kr &= 0x10;
		kl |= kr;
	}
	o_bop((unsigned)(kl << 24) | op);
	ts[nts].flags = 0;
	ts[nts].ptr = 0;
	ts[nts].tag = 0;
	ts[nts++].kind = kl;
}
