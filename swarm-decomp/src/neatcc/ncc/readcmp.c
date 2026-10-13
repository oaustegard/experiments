#include <string.h>
#include "fn_tok_get.h"
#include "fn_tok_see.h"
#include "readadd.h"
#include "shift.h"
#include "cmp.h"
#include "readcmp.h"

static void readcmp(void)
{
	readadd();
	for (;;) {
		if (strcmp("<<", tok_see()) == 0) {
			tok_get();
			shift(0x20);
			continue;
		}
		if (strcmp(">>", tok_see()) == 0) {
			tok_get();
			shift(0x21);
			continue;
		}
		break;
	}
	for (;;) {
		if (strcmp("<", tok_see()) == 0) {
			tok_get();
			cmp(0x80);
			continue;
		}
		if (strcmp(">", tok_see()) == 0) {
			tok_get();
			cmp(0x85);
			continue;
		}
		if (strcmp("<=", tok_see()) == 0) {
			tok_get();
			cmp(0x84);
			continue;
		}
		if (strcmp(">=", tok_see()) == 0) {
			tok_get();
			cmp(0x81);
			continue;
		}
		break;
	}
}
