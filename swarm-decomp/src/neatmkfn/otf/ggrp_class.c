#include <stdlib.h>
#include "ggrp_find.h"
#include "ggrp_make.h"
#include "intcmp.h"

static int
ggrp_class(int *g, int *cls, int n, int v)
{
	int *tmp;
	int i, m, r;

	tmp = malloc(n * sizeof *tmp);
	m = 0;
	for (i = 0; i < n; i++)
		if (cls[i] == v)
			tmp[m++] = g[i];
	qsort(tmp, m, sizeof *tmp, intcmp);
	r = ggrp_find(tmp, m);
	if (r >= 0)
		goto done;
	r = ggrp_make(tmp, m);
done:
	free(tmp);
	return r;
}
