#include "libc.h"

static char *tok_quotes(char *s)
{
	if (s == NULL)
		s = "";
	else if (*s == '"') {
		s[strlen(s) - 1] = '\0';
		return s + 1;
	}
	return s;
}
