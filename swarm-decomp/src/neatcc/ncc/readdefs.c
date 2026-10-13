#include <string.h>
#include "decl.h"
#include "type.h"
#include "fn_tok_see.h"
#include "fn_tok_get.h"
#include "basetype.h"
#include "readname.h"
#include "readdefs.h"

static int readdefs(void (*fn)(long, struct decl *, int), long local)
{
	struct type base;
	int sc, ret = 1;

	if (basetype(&base, &sc))
		return ret;
	ret = 0;
	if (strcmp(tok_see(), ";") == 0)
		return ret;
	if (strcmp(tok_see(), "{") == 0)
		return ret;
	do {
		struct decl d = {{0}};
		if (readname(&d.t, d.name, &base))
			return ret;
		fn(local, &d, sc);
	} while (strcmp(tok_see(), ",") == 0 && (tok_get(), 1));
	return ret;
}
