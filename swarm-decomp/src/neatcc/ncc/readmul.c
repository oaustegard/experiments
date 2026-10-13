#include <string.h>
#include "fn_tok_see.h"
#include "fn_tok_get.h"
#include "ts_binop.h"
#include "readpre.h"
#include "readmul.h"

static void readmul(void)
{
	char *t;

	readpre();
	for (;;) {
		while (strcmp("*", tok_see()) == 0) {
			tok_get();
			readpre();
			ts_binop(0x40);
		}
		t = tok_see();
		if (strcmp(t, "/") == 0) {
			tok_get();
			readpre();
			ts_binop(0x41);
			continue;
		}
		t = tok_see();
		if (strcmp(t, "%") != 0)
			return;
		tok_get();
		readpre();
		ts_binop(0x42);
	}
}
