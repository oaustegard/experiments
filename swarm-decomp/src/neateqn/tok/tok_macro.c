void tok_macro(void)
{
	struct sbuf sb;
	char name[32];
	int c, d;

	tok_preview(name, 0);
	sbuf_init(&sb);
	c = src_next();
	while (c > 0 && isspace(c))
		c = src_next();
	d = src_next();
	while (d > 0 && c != d) {
		sbuf_add(&sb, d);
		d = src_next();
	}
end:
	src_define(name, sbuf_buf(&sb));
	sbuf_done(&sb);
}
