static void opassign(int op, char *p)
{
	struct type t;
	int k;

	t = ts[nts - 1];
	o_tmpcopy();
	ts[nts++] = t;
	readexpr();
	if (op == 0x10 || op == 0x11)
		ts_addop(op);
	else
		ts_binop(op);
	k = 8;
	if (ts[nts - 2].ptr == 0)
		k = ts[nts - 2].kind;
	o_assign(k);
	nts--;
	ts_de(0);
}
