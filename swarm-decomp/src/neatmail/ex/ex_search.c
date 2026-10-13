#include "fn_mbox_len.h"
#include "vars.h"
#include "ex_keyword.h"
#include "ex_match.h"
#include "ex_search.h"

static int ex_search(char *s)
{
	int i, d;

	i = pos;
	d = s[0] == '/' ? 1 : -1;
	if (ex_keyword(s))
		return 1;
	i += d;
	while (i >= 0 && mbox_len(mbox) > i) {
		if (!ex_match(i))
			return i;
		i += d;
		if (i == -1)
			break;
	}
	return pos;
}
