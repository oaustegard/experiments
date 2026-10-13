#include "intcmp.h"

static int
intcmp(const void *a, const void *b)
{
	return *(const int *)a - *(const int *)b;
}
