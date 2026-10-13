#include <stdlib.h>
#include "rnode.h"
#include "rnode_seq.h"

static struct rnode *rnode_parse(char **s)
{
	struct rnode *l = rnode_seq(s);
	struct rnode *r, *n;

	if (**s != '|')
		return l;
	(*s)++;
	r = rnode_parse(s);
	if (!r)
		return l;
	n = calloc(sizeof(struct rnode), 1);
	n->type = '|';
	n->left = l;
	n->right = r;
	n->min = 1;
	n->max = 1;
	return n;
}
