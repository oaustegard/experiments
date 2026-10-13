void box_merge(struct box *b, struct box *m, int c)
{
	char *s;

	if (*sbuf_buf(&m->buf) == 0)
		return;
	box_beforeput(b, m->first, c);
	box_toreg(b);
	s = box_toreg(m);
	sbuf_append(&b->buf, s);
	if (b->sreg_id)
		printf(".as %s \"%s\n", sregname(b->sreg_id), s);
	if (b->first == 0)
		b->first = m->first;
	if (m->nitems == 1) {
		box_afterput(b, m->flags);
		return;
	}
	b->nitems += m->nitems;
	b->flags = m->flags;
}
