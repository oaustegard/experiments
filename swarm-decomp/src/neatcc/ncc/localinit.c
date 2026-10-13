static void localinit(long *loc, int val, struct type *t)
{
	long off = *loc;

	if (t->flags & 1) {
		if (tok_grp() == '"') {
			struct array *a = (struct array *)arrays + t->arr;

			if (a->t.ptr == 0 && a->t.flags == 0 && (a->t.kind & 0xf) == 1) {
				char *s = tok_get() + 1;
				int len = tok_len();
				int m = len - 2;

				o_local(off);
				if (val) {
					o_num(val);
					o_bop(0x10);
				}
				o_sym(tmp_str(s, m));
				o_num(len - 1);
				o_memcpy();
				o_tmpdrop(1);
				return;
			}
		}
	}
	o_local(off);
	if (val) {
		o_num(val);
		o_bop(0x10);
	}
	ts[nts++] = *t;
	readexpr();
	doassign();
	nts--;
	o_tmpdrop(1);
}
