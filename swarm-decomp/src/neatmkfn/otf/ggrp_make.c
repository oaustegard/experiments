#include <stdlib.h>
#include <string.h>
#include "ggrp_find.h"
#include "ggrp_make.h"
#include "mkfn_byname.h"
#include "mkfn_proto.h"

static int
ggrp_make(int *g, int n)
{
	static char str[16];
	int k, i;

	k = ggrp_find(g, n);
	if (k >= 0)
		return k;
	k = ggrp_n++;
	ggrp_g[k] = malloc(n * sizeof *g);
	ggrp_len[k] = n;
	if (n > 0) {
		memcpy(ggrp_g[k], g, n * sizeof *g);
		printf("ggrp %d %d", k, n);
		for (i = 0; i < n; i++) {
			printf(" %s", (!mkfn_byname && strlen(glyph_name[g[i]]) > 3
			    ? (sprintf(str, "%d", g[i]), str) : glyph_name[g[i]]));
		}
	} else
		printf("ggrp %d %d", k, n);
	putchar('\n');
	return k;
}
