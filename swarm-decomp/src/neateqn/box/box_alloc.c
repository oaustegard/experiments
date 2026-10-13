struct box *box_alloc(int type, int flags, int col)
{
	struct box *b = calloc(sizeof(struct box), 1);

	sbuf_init(&b->buf);
	b->type = type;
	b->nitems = 0;
	b->col = col;
	if (flags)
		b->flags = flags;
	return b;
}
