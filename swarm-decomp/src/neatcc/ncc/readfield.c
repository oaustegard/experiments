static void readfield(void)
{
	struct type t;
	char *name;
	struct decl *f;

	nts--;
	t = ts[nts];
	if (t.flags & 1) {
		if (t.ptr == 0) {
			t = ((struct array *)arrays)[t.arr].t;
			t.ptr++;
		}
	}
	name = tok_get();
	f = field_find(&t, name);
	if (!f)
		err("unknown field <%s>\n", name);
	if (*(long *)f->loc) {
		o_num(*(long *)f->loc);
		o_bop(0x10);
	}
	ts[nts] = f->t;
	ts[nts].tag = 1;
	nts++;
}
