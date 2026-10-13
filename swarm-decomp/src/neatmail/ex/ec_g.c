#include <ctype.h>
#include "ex_keyword.h"
#include "ex_match.h"
#include "ex_exec.h"
#include "vars.h"
#include "ec_g.h"

static int ec_g(char *s, int inv)
{
	char *p;
	unsigned char c;
	int r;

	while (isspace((unsigned char)*s))
		s++;
	c = *s;
	if (c != '/')
		return c;
	r = ex_keyword(s);
	if (r)
		return r;
	for (p = s + 1; (c = *p) && c != '/'; p++)
		;
	if (!kwd[0])
		return c;
	if (*p != '/')
		return c;
	if ((ex_match(pos) != 0) == (inv ^ 1))
		return c;
	return ex_exec(p + 1);
}
