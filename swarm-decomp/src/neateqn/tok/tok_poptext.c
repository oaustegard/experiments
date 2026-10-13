char *tok_poptext(int c)
{
	if (tok[0]) {
		for (;;) {
			if (tok_curtype != 1)
				break;
			tok_read();
			if (!tok[0])
				break;
		}
	}
	tok_prev[0] = 0;
	for (;;) {
		strcat(tok_prev, tok);
		tok_read();
		if (!tok[0] || tok_curtype == 3)
			break;
		if (c) {
			if (def_chopped((unsigned char)tok[0]))
				break;
		} else if (strchr("^~{}(),\"\n\t =:|.+-*/\\,()[]<>!", (unsigned char)tok[0])) {
			break;
		}
	}
	return tok_prev[0] ? tok_prev : 0;
}
