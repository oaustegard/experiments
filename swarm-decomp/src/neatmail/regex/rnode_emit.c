#include "rnode.h"
#include "regex_t.h"
#include "rnode_emit.h"
#include "rnode_emitnorep.h"

static void rnode_emit(struct rnode *n, struct rprog *prog)
{
	int i, j, last, idx, k;

	if (!n)
		return;
	if (!n->min && !n->max)
		return;
	if (n->min == 1 && n->max == 1) {
		rnode_emitnorep(n, prog);
		return;
	}
	for (i = 0; i < n->min; i++)
		rnode_emitnorep(n, prog);
	if (n->max > n->min) {
		last = -1;
		for (j = n->min; j < n->max; j++) {
			idx = prog->n;
			prog->n = idx + 1;
			prog->slots[idx].flag = 'f';
			prog->slots[idx].link = last;
			last = idx;
			rnode_emitnorep(n, prog);
		}
		for (k = last; k >= 0; ) {
			int next = prog->slots[k].link;
			prog->slots[k].link = prog->n;
			k = next;
		}
	}
	if (n->max < 0) {
		idx = prog->n;
		prog->n = idx + 1;
		prog->slots[idx].flag = 'f';
		rnode_emitnorep(n, prog);
		k = prog->n;
		prog->n = k + 1;
		prog->slots[k].flag = 'j';
		prog->slots[k].link = idx;
		prog->slots[idx].link = k + 1;
	}
}
