#include <stdlib.h>
#include <string.h>
#include "rnode.h"
#include "regex_t.h"
#include "rnode_emit.h"
#include "rnode_emitnorep.h"

static void rnode_emitnorep(struct rnode *n, struct rprog *prog)
{
	int idx, len;
	struct rslot *s;

	if (n->type == '|') {
		idx = prog->n;
		prog->n = idx + 1;
		prog->slots[idx].flag = 'f';
		rnode_emit(n->left, prog);
		len = prog->n;
		prog->n = len + 1;
		prog->slots[len].flag = 'j';
		prog->slots[idx].link = len + 1;
		rnode_emit(n->right, prog);
		prog->slots[len].link = prog->n;
	}
	if (n->type == 'c') {
		rnode_emit(n->left, prog);
		rnode_emit(n->right, prog);
	}
	if (n->type == '(') {
		idx = prog->n;
		prog->n = idx + 1;
		s = &prog->slots[idx];
		s->c18 = n->c28 * 2;
		s->flag = 'm';
		rnode_emit(n->left, prog);
		idx = prog->n;
		prog->n = idx + 1;
		s = &prog->slots[idx];
		s->flag = 'm';
		s->c18 = n->c28 * 2 + 1;
	}
	if (!n->type) {
		idx = prog->n;
		prog->n = idx + 1;
		s = &prog->slots[idx];
		s->flag = 0;
		s->c00 = n->c00;
		s->str = NULL;
		if (n->str) {
			len = strlen(n->str) + 1;
			s->str = malloc(len);
			memcpy(s->str, n->str, len);
		}
		return;
	}
}
