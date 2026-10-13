char *box_toreg(struct box *b)
{
	char *s;

	if (b->sreg_id == 0) {
		b->sreg_id = sregmk();
		s = sbuf_buf(&b->buf);
		printf(".ds %s \"%s\n", sregname(b->sreg_id), s);
	}
	return sreg(b->sreg_id);
}
