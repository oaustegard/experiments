#include <string.h>
#include "fn_tok_get.h"
#include "fn_tok_see.h"
#include "ts_addop.h"
#include "readmul.h"
#include "readadd.h"

static void readadd(void)
{
	char *t;

	readmul();
	for (;;) {
		while (strcmp("+", tok_see()) == 0) {
			tok_get();
			readmul();
			ts_addop(0x10);
		}
		t = tok_see();
		if (strcmp("-", t) != 0)
			return;
		tok_get();
		readmul();
		ts_addop(0x11);
	}
}
