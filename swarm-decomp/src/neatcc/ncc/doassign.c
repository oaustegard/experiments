static void doassign(void)
{
	struct type t;
	int k;

	t = ts[nts - 1];
	if (t.ptr == 0 && (t.flags & 2)) {
		nts--;
		o_num(type_totsz(&t));
		o_memcpy();
		return;
	}
	ts_de(1);
	nts--;
	k = 8;
	if (ts[nts - 1].ptr == 0)
		k = ts[nts - 1].kind;
	o_assign(k);
	ts_de(0);
}
