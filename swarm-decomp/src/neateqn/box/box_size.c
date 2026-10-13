int box_size(struct box *b, const char *s)
{
	int t = b->type;

	if (!s || !*s)
		return t;
	if (!b->nreg_set) {
		b->nreg_set = 1;
		b->type = nregmk();
	}
	if (*s == '+' || *s == '-')
		printf(".nr %s %s%s\n", nregname(b->type), nreg(t), s);
	else
		printf(".nr %s %s\n", nregname(b->type), s);
	return b->type;
}
