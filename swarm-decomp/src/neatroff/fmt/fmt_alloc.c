#include <string.h>
#include "fmt.h"
#include "fn_xmalloc.h"

void *fmt_alloc(void)
{
	struct fmt *f;

	f = xmalloc(sizeof(struct fmt));
	memset(f, 0, sizeof(struct fmt));
	return f;
}
