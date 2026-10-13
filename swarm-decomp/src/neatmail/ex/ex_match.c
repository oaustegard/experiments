#include <stdlib.h>
#include <string.h>
#include "fn_mbox_get.h"
#include "fn_msg_get.h"
#include "fn_hdrlen.h"
#include "ex_regex.h"
#include "vars.h"
#include "ex_match.h"

static int ex_match(int i)
{
	char *s, *p, *buf;
	long n;
	int len, r;

	if (mbox_get(mbox, i, &s, &n))
		return 1;
	p = msg_get(s, n, kwd_hdr);
	if (!p)
		return 1;
	len = hdrlen(p, s + n - p);
	buf = malloc(len + 1);
	memcpy(buf, p, len);
	buf[len] = 0;
	r = regexec(&kwd_re, buf, 0, 0, 0);
	free(buf);
	return r != 0;
}
