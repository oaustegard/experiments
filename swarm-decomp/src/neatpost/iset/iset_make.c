#include <stdlib.h>
#include "iset.h"
#include "fn_iset_make.h"

struct iset *iset_make(void)
{
	struct iset *s;

	s = calloc(sizeof(struct iset), 1);
	iset_extend(s, 0x400);
	return s;
}
