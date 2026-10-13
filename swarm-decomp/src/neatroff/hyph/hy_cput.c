#include "hyph_ext.h"
#include "fn_hy.h"
#include "fn_memset.h"

int hy_cput(char *d, char *s)
{
	if (!*s || !s[1] || utf8one(s))
		strcpy(d, s);
	else if (*s == '\\')
		strcpy(d, s);
	else if (!s[2])
		snprintf(d, 32, "\\[%s]", s);
	return strlen(d);
}
