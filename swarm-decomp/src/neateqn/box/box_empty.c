int box_empty(struct box *b)
{
	return *sbuf_buf(&b->buf) == 0;
}
