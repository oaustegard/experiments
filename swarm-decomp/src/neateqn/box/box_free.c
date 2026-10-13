void box_free(struct box *b)
{
	if (b->sreg_id)
		sregrm(b->sreg_id);
	if (b->nreg_set)
		nregrm(b->type);
	sbuf_done(&b->buf);
	free(b);
}
