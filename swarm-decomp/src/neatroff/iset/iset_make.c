#include "iset.h"
#include "fn_xmalloc.h"

void *iset_make(void)
{
	struct iset *s = xmalloc(sizeof(struct iset));

	*s = (struct iset){ 0 };
	iset_extend(s, 1024);
	return s;
}
