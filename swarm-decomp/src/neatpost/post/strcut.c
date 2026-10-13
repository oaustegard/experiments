static char *strcut(char *dst, char *src)
{
	char *q;
	unsigned char c;

	q = src;
	c = *q;

	for (;;) {
		if (c == ' ') {
			c = *++q;
			continue;
		}
		if (c == '\n') {
			c = *++q;
			continue;
		}
		break;
	}

	if (c == '"') {
		q = q + 1;
		while (*q != 0) {
			if (*q == '"') {
				if (q[1] != '"') {
					*dst = 0;
					return q + 1;
				}
				q++;
			}
			*dst++ = *q;
			q++;
		}
		*dst = 0;
		return q;
	}
	while (c != 0 && c != ' ') {
		if (c == '\n')
			break;
		*dst++ = c;
		c = *++q;
	}
	*dst = 0;
	return q;
}
