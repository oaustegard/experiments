void box_puttext(struct box *b, int c, const char *fmt, ...)
{
	char buf[1000];
	va_list ap;

	va_start(ap, fmt);
	vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);
	box_beforeput(b, c, 0);
	if (!(b->flags & 0x100) && (c & 0x100)) {
		sbuf_append(&b->buf, "\\,");
		if (b->sreg_id)
			printf(".as %s \"%s\n", sregname(b->sreg_id), "\\,");
	}
	sbuf_append(&b->buf, buf);
	if (b->sreg_id)
		printf(".as %s \"%s\n", sregname(b->sreg_id), buf);
	box_afterput(b, c);
}
