void box_putf(struct box *b, const char *fmt, ...)
{
	char buf[1000];
	va_list ap;

	va_start(ap, fmt);
	vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);
	sbuf_append(&b->buf, buf);
	if (b->sreg_id)
		printf(".as %s \"%s\n", sregname(b->sreg_id), buf);
}
