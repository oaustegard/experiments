static void box_italiccorrection(struct box *b)
{
	if (b->nitems != 0 && (b->flags & 0x100)) {
		sbuf_append(&b->buf, "\\/");
		if (b->sreg_id != 0)
			printf(".as %s \"%s\n", sregname(b->sreg_id), "\\/");
	}
	b->flags &= ~0x100;
}
