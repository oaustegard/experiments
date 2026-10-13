void box_move(struct box *b, int n, int m)
{
	if (n)
		box_putf(b, "\\v'%du*%sp/100u'", n, nreg(b->type));
	if (m)
		box_putf(b, "\\h'%du*%sp/100u'", m, nreg(b->type));
}
