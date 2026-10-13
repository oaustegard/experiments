#include <string.h>
#include "fn_tok_see.h"
#include "fn_tok_get.h"
#include "ts_binop.h"
#include "readeq.h"
#include "readxor.h"

static void readxor(void)
{
	readeq();
	for (;;) {
		if (strcmp("&", tok_see()) != 0)
			break;
		tok_get();
		readeq();
		ts_binop(0x12);
	}
	while (strcmp("^", tok_see()) == 0) {
		tok_get();
		readeq();
		while (strcmp("&", tok_see()) == 0) {
			tok_get();
			readeq();
			ts_binop(0x12);
		}
		ts_binop(0x14);
	}
}
