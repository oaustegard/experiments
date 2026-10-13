#include <ctype.h>
#include "fn_tok_see.h"
#include "tok_grp.h"

static int tok_grp(void)
{
	int c = (unsigned char)*tok_see();

	if (c == '"')
		return c;
	if (c == '\'')
		return '0';
	if (isdigit(c))
		return '0';
	if (c == '_')
		return 'a';
	return isalpha(c) ? 'a' : 0;
}
