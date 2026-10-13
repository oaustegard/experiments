static int tok_next(void)
{
	int c;

	if (!tok_eqen && !tok_line)
		return 0;
	c = src_next();
	if (tok_eqen && c == '\n') {
		if (tok_req(0))
			tok_eqen = 0;
	}
	if (!tok_line)
		return c;
	if (!src_top())
		return c;
	if (c == eqn_end) {
		tok_line = 0;
		return 0;
	}
	return c;
}
