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
	char *s;
	int k, i;
	int *p;

	k = ggrp_find(g, n);
	if (k >= 0)
		return k;
	k = ggrp_n++;
	ggrp_g[k] = malloc(n * sizeof *g);
	ggrp_len[k] = n;
	if (n > 0) {
		memcpy(ggrp_g[k], g, n * sizeof *g);
		printf("ggrp %d %d", k, n);
		for (i = 0, p = g; i < n; i++, p++) {
			s = glyph_name[*p];
			if (!mkfn_byname && strlen(s) > 3) {
				sprintf(str, "%d", *p);
				s = str;
			}
			printf(" %s", s);
		}
	} else
		printf("ggrp %d %d", k, n);
	putchar('\n');
	return k;
}
