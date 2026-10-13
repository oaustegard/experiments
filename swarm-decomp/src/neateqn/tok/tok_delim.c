void tok_delim(void)
{
	char s[32];
	int beg = 0, end = 0;

	tok_preview(s, 0);
	if (strcmp(s, "off")) {
		beg = s[0];
		end = s[1];
	}
	eqn_end = end;
	eqn_beg = beg;
}
