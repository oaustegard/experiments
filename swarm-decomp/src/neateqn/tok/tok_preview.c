static void tok_preview(char *s, int f)
{
	int c, n;

	c = src_next();
	if (c > 0)
		goto cont;
	goto done;
cont:
	if (def_chopped(c)) {
		src_back(c);
		s += readchar(s);
		goto end;
	}
	n = 0;
	while (!def_chopped(c)) {
		if (!f || c != '(') {
			if (tok_line && src_top() && c == eqn_end)
				break;
			src_back(c);
			n += readchar(s + n);
			c = src_next();
			if (c <= 0)
				break;
		} else
			break;
	}
	s += n;
done:
	src_back(c);
end:
	*s = 0;
}
