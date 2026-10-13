#include <stdlib.h>
#include "rnode.h"
#include "regex_t.h"
#include "rnode_emit.h"
#include "rnode_free.h"
#include "rnode_grpnum.h"
#include "rnode_parse.h"

int regcomp(regex_t *preg, char *pattern, int cflags)
{
	char *s = pattern;
	struct rnode *n, *p;
	struct rprog *prog;
	struct rslot *slots;
	int count, ngrp, k, g, ret;

	n = rnode_parse(&s);
	count = rnode_count(n);
	ret = 1;
	if (n) {
		ngrp = 1;
		for (p = n; p; p = p->right) {
			k = 0;
			g = ngrp;
			if (p->type == '(') {
				p->c28 = ngrp;
				g = ngrp + 1;
				k = 1;
			}
			k += rnode_grpnum(p->left, g);
			ngrp += k;
		}
		prog = malloc(sizeof(struct rprog));
		prog->cflags = 0;
		slots = calloc((size_t)(count + 3) * sizeof(struct rslot), 1);
		prog->n = 1;
		prog->slots = slots;
		prog->slots[0].flag = 'm';
		prog->slots[0].c18 = 0;
		rnode_emit(n, prog);
		k = prog->n;
		prog->slots[k].flag = 'm';
		prog->slots[k].c18 = 1;
		prog->n = k + 2;
		prog->slots[k + 1].flag = 'q';
		rnode_free(n);
		prog->cflags = cflags;
		preg->prog = prog;
		ret = 0;
	}
	return ret;
}
