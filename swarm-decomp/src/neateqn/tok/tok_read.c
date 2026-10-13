static int tok_read(void)
{
	int c, c2, c3, c4, n, i, x, nargs, depth, quote;
	char *p;
	char *args[10];
	struct sbuf sb[10];
	struct sbuf *sbp;

	for (;;) {
		tok[0] = 0;
		c = tok_next();
		if (c <= 0)
			return 0;
		tok_prevsep = tok_cursep;
		tok_cursep = def_chopped(c);
		if (tok_cursep)
			tok_prevsep = 1;
		if (c == ' ' || c == '\n') {
			c2 = tok_next();
			while (c2 > 0 && (c2 == ' ' || c2 == '\n'))
				c2 = tok_next();
			if (tok_eqen || tok_line)
				src_back(c2);
			tok_curtype = 1;
			tok[0] = ' ';
			tok[1] = 0;
			return 0;
		}
		if (c == '\t') {
			tok_curtype = 2;
			tok[0] = '\t';
			tok[1] = 0;
			return 0;
		}
		if (!tok_prevsep)
			goto punct;
		if (c == '$') {
			c2 = tok_next();
			if ((unsigned)(c2 - '1') <= 8 && !src_arg(c2 - '0')) {
				tok_cursep = 1;
				continue;
			}
			if (tok_eqen || tok_line)
				src_back(c2);
		}
		if (tok_eqen || tok_line)
			src_back(c);
		tok_preview(tok, 0);
		for (i = 0; i < (int)(sizeof kwds / sizeof kwds[0]); i++) {
			if (!strcmp(kwds[i], tok)) {
				tok_curtype = 3;
				tok_cursep = 1;
				return 0;
			}
		}
		n = strlen(tok);
		while (n > 0) {
			n--;
			src_back((unsigned char)tok[n]);
		}
		for (i = 0; i < 10; i++)
			args[i] = 0;
		tok_preview(tok, 1);
		if (src_macro(tok)) {
			c = src_next();
			src_back(c);
			if (c != '(') {
				src_expand(0, args);
				tok_cursep = 1;
				continue;
			}
			src_next();
			nargs = 0;
			sbp = sb;
			for (;;) {
				sbuf_init(sbp);
				nargs++;
				c = src_next();
				if (c <= 0)
					break;
				depth = 0;
				quote = 0;
				for (;;) {
					if (!quote && !depth) {
						if (c == ',' || c == ')')
							break;
						sbuf_add(sbp, c);
						if (c == '(')
							depth = -1;
						else if (c == '"')
							quote = 1;
						else if (c == '\\')
							goto escape;
					} else {
						sbuf_add(sbp, c);
						if (c == ')' && !quote)
							depth++;
						else if (c == '(' && !quote)
							depth--;
						else if (c == '"')
							quote = !quote;
						else if (c == '\\')
							goto escape;
					}
					goto next;
escape:
					c2 = src_next();
					sbuf_add(sbp, c2);
					if (c2 == '*' || c2 == 'n') {
						c2 = src_next();
						sbuf_add(sbp, c2);
					}
					if (c2 == '(') {
						c3 = src_next();
						sbuf_add(sbp, c3);
						c4 = src_next();
						sbuf_add(sbp, c4);
					} else if (c2 == '[') {
						do {
							c3 = src_next();
							sbuf_add(sbp, c3);
						} while (c3 > 0 && c3 != ']');
					}
next:
					c = src_next();
					if (c <= 0)
						goto args_done;
				}
				sbp++;
				if (c != ',')
					break;
				if (nargs == 10)
					break;
			}
args_done:
			for (i = 0; i < nargs; i++)
				args[i] = sbuf_buf(&sb[i]);
			src_expand(0, args);
			for (i = 0; i < nargs; i++)
				sbuf_done(&sb[i]);
			tok_cursep = 1;
			continue;
		}
		n = strlen(tok);
		while (n > 0) {
			n--;
			src_back((unsigned char)tok[n]);
		}
		c = tok_next();
punct:
		if (!strchr("^~{}(),\"\n\t =:|.+-*/\\,()[]<>!", c))
			goto word;
		tok[0] = c;
		if (c == '\\') {
			c2 = tok_next();
			if (c2 == '(') {
				tok[1] = '(';
				tok[2] = tok_next();
				tok[3] = tok_next();
				tok[4] = 0;
			} else if (c2 == '[') {
				p = tok + 1;
				c3 = c2;
				for (;;) {
					if (p < tok + 998)
						*p++ = c3;
					c3 = tok_next();
					if (c3 == 0 || c3 == ']')
						break;
				}
				*p = ']';
				p[1] = 0;
			} else {
				tok[1] = 0;
			}
			goto classify;
		}
		if (c == '"') {
			c3 = tok_next();
			p = tok + 1;
			while (c3 > 0 && c3 != '"') {
				if (c3 == '\\') {
					c4 = tok_next();
					if (c4 == '"')
						c3 = '"';
					else if (tok_eqen || tok_line)
						src_back(c4);
				}
				if (p < tok + 998)
					*p++ = c3;
				c3 = tok_next();
			}
			*p = '"';
			p[1] = 0;
			goto classify;
		}
		c2 = tok_next();
		x = (c << 8) | c2;
		if (x == 0x3c2d || x == 0x2d3e || x == 0x213d || x == 0x2d2b || x == 0x3d3d ||
		    x == 0x3c3c || x == 0x3c3d || x == 0x3e3d || x == 0x3e3e || x == 0x3a3d) {
			tok[1] = c2;
			tok[2] = 0;
			goto classify;
		}
		if (tok_eqen || tok_line)
			src_back(c2);
		tok[1] = 0;
		goto classify;
word:
		p = tok + 1;
		if (!(c & 0x80) || !(c & 0x40)) {
			tok[1] = 0;
			goto classify;
		}
		if (!(c & 0x20))
			n = 1;
		else if (!(c & 0x10))
			n = 2;
		else {
			if (c & 8) {
				tok[1] = 0;
				goto classify;
			}
			n = 3;
		}
		for (i = 0; i < n; i++)
			p[i] = tok_next();
		p[n] = 0;
		tok_curtype = char_type((const unsigned char *)tok);
		return 0;
classify:
		tok_curtype = char_type((const unsigned char *)tok);
		return 0;
	}
}
