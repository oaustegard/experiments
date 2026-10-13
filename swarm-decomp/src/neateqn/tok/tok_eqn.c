int tok_eqn(void)
{
	struct sbuf sb;
	int c;
	unsigned char *s;
	unsigned char *p;
	unsigned char *q;

	tok_cursep = 1;
	sbuf_init(&sb);
	c = src_next();
	while (c > 0) {
		if (c == eqn_beg) {
			puts(".eo");
			s = (unsigned char *)sbuf_buf(&sb);
			printf(".%s %s \"%s\n", tok_part ? "as" : "ds", "10", s);
			sbuf_done(&sb);
			puts(".ec");
			tok_part = 1;
			tok_line = 1;
			return 0;
		}
		sbuf_add(&sb, c);
		if (c == '\n') {
			if (tok_part) {
				printf(".lf %d\n", src_lineget());
				s = (unsigned char *)sbuf_buf(&sb);
				printf("\\*%s%s", escarg("10"), s);
				tok_part = 0;
			} else {
				printf("%s", sbuf_buf(&sb));
				s = (unsigned char *)sbuf_buf(&sb);
				if (*s == '.') {
					p = s + 1;
					while (isspace(*p))
						p++;
					if (*p == 'l' && p[1] == 'f') {
						q = p + 2;
						while (isspace(*q))
							q++;
						if (isdigit(*q))
							src_lineset(strtol((char *)q, 0, 10));
					}
				}
				s = (unsigned char *)sbuf_buf(&sb);
				if (*s == '.') {
					p = s + 1;
					while (isspace(*p))
						p++;
					if (*p == 'E' && p[1] == 'Q' && tok_req(0) == 0) {
						tok_eqen = 1;
						sbuf_done(&sb);
						return 0;
					}
				}
			}
			sbuf_cut(&sb, 0);
		}
		c = src_next();
	}
	sbuf_done(&sb);
	return 1;
}
