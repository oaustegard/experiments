int tok_jmp(char *s)
{
	if (!tok[0])
		return 1;
	while (tok_curtype == 1) {
		strcpy(tok_prev, tok);
		tok_read();
		if (!tok[0])
			break;
	}
	if (!tok[0])
		return 1;
	if (s[1] != 0 || !strchr("{}~^\t", s[0])) {
		if (tok_curtype != 3)
			return 1;
		if (strcmp(s, tok) != 0)
			return 1;
	} else if (strcmp(s, tok) != 0)
		return 1;
	strcpy(tok_prev, tok);
	tok_read();
	return 0;
}
