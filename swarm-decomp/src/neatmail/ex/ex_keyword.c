#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include "fn_sbuf.h"
#include "ex_regex.h"
#include "fn_regfree.h"
#include "vars.h"
#include "ex_keyword.h"

static int ex_keyword(char *s)
{
	struct sbuf *sb;
	char *p;
	char c;
	int n;

	p = s;
	sb = sbuf_make();
	while ((c = *++p) && c != *s) {
		if (c == '\\' && p[1])
			p++;
		sbuf_chr(sb, (unsigned char)*p);
	}
	if (sbuf_len(sb) != 0 && strcmp(kwd, sbuf_buf(sb)) != 0) {
		if (kwd[0])
			regfree(&kwd_re);
		snprintf(kwd, 512, "%s", sbuf_buf(sb));
		if (regcomp(&kwd_re, kwd, 5) != 0)
			kwd[0] = 0;
		if (kwd[0] == '^' && isalpha((unsigned char)kwd[1]) && (p = strchr(kwd, ':')) != 0) {
			n = p - kwd;
			memcpy(kwd_hdr, kwd + 1, n);
			kwd_hdr[n] = 0;
		} else
			strcpy(kwd_hdr, "subject:");
	}
	sbuf_free(sb);
	return kwd[0] == 0;
}
