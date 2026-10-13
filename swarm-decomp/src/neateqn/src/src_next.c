int
src_next(void)
{
	struct esrc *cur, *prev;
	int i, c;

	for (;;) {
		cur = esrc;
		if (cur->nback != 0) {
			cur->nback--;
			return cur->back[cur->nback];
		}
		prev = cur->prev;
		if (prev == NULL) {
			c = fgetc(stdin);
			if (c == '\n')
				lineno++;
			return c;
		}
		if (cur->text[cur->pos] != '\0') {
			cur->pos++;
			return (unsigned char)cur->text[cur->pos - 1];
		}
		for (i = 0; i < 10; i++)
			free(cur->arg[i]);
		free(cur->text);
		free(cur);
		esrc = prev;
		esrc_depth--;
	}
}
