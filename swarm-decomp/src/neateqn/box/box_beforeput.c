static void box_beforeput(struct box *b, int c, int d)
{
	int it = c & 0x100;
	int cl, fl, sp, k, r12;

	if (b->nitems == 0)
		goto tail;
	if ((b->col >> 4) != 0)
		goto nospace;
	cl = c & 0xf0;
	fl = b->flags & 0xf0;
	if (fl == 0 || cl == 0)
		goto nospace;
	k = spacing[(fl >> 4) - 1][(cl >> 4) - 1];
	sp = e_thickspace;
	if (k != 3) {
		if (k == 2) {
			sp = e_mediumspace;
		} else {
			sp = e_thinspace;
			if (k == 0)
				goto nospace;
		}
	}
	r12 = c != 0x14 && sp != 0;
	if (!it)
		box_italiccorrection(b);
	if (!r12)
		goto tail;
	if (b->flags == 0x14)
		goto tail;
	box_italiccorrection(b);
	if (d) {
		int bc = def_brcost(b->flags & 0xf0);

		box_putf(b, "\\s[\\En(.s*%du*%sp/100u/\\w' 'u]\\j'%d' \\s0", sp, nreg(b->type), bc);
	} else {
		box_putf(b, "\\h'%du*%sp/100u'", sp, nreg(b->type));
	}
	goto tail;
nospace:
	if (!it)
		box_italiccorrection(b);
tail:
	if (b->wname != 0) {
		printf(".nr %s 0\\w'%s'\n", b->wname, box_toreg(b));
		b->wname = 0;
	}
}
