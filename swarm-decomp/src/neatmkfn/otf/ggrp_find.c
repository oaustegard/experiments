#include "ggrp_find.h"
#include "ggrp_make.h"

static int
ggrp_find(int *g, int n)
{
	int i, j;

	for (i = 0; i < ggrp_n; i++) {
		if (ggrp_len[i] != n)
			continue;
		for (j = 0; j < n; j++)
			if (g[j] != ggrp_g[i][j])
				break;
		if (j == n)
			return i;
	}
	return -1;
}
