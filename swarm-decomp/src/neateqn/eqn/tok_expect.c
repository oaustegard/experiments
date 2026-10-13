#include "libc.h"
#include "fn_tok_get.h"
#include "fn_tok_jmp.h"

static void tok_expect(const char *s)
{
	if (tok_jmp((char *) s)) {
		fprintf(stderr, "neateqn: expected %s bot got %s\n", s, tok_get());
		exit(1);
	}
}
