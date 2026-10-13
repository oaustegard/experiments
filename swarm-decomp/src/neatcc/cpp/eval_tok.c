#include <ctype.h>
#include "cpp_protos.h"
#include "cpp_etok.h"

static int eval_tok(void)
{
	int start = ecur;
	int i = start;
	int skipped = 0;
	char c;
	int key;
	char *d;
	const char *const *p;

	if (start >= elen)
		return -1;
	c = ebuf[i];
	for (;;) {
		while (i < elen && isspace(c)) {
			i++;
			skipped = 1;
			c = ebuf[i];
		}
		if (c != '/' || ebuf[i + 1] != '*')
			break;
		while (i < elen) {
			if (ebuf[i - 2] == '*' && ebuf[i - 1] == '/')
				break;
			i++;
			skipped = 1;
		}
		if (i >= elen)
			break;
		c = ebuf[i];
	}
	if (skipped)
		ecur = i;
	else
		i = start;
	if (i >= elen)
		return -1;

	if (isalpha(c) || c == '_') {
		d = etok;
		*d++ = c;
		for (i++; isalnum(ebuf[i]) || ebuf[i] == '_'; i++)
			*d++ = ebuf[i];
		*d = 0;
		ecur = i;
		return 0x100;
	}
	if (isdigit(c)) {
		d = etok;
		*d++ = c;
		for (i++; isdigit(ebuf[i]); i++)
			*d++ = ebuf[i];
		ecur = i;
		while (tolower(ebuf[i]) == 'u' || tolower(ebuf[i]) == 'l')
			i++;
		ecur = i;
		*d = 0;
		return 0x101;
	}
	key = c << 16 | ebuf[i + 1] << 8;
	for (p = tok2; p < tok2 + 8; p++) {
		if (((*p)[0] << 16 | (*p)[1] << 8) == key) {
			ecur = i + 2;
			return key;
		}
	}
	ecur = i + 1;
	return c;
}
