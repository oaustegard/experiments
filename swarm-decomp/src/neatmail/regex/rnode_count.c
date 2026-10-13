#include "rnode.h"

static int rnode_count(struct rnode *n)
{
	int c;

	if (!n)
		return 0;
	if (n->type == 'c') {
		c = rnode_count(n->left);
		c += rnode_count(n->right);
	} else if (n->type == '|') {
		c = rnode_count(n->left) + rnode_count(n->right) + 2;
	} else {
		c = 1;
		if (n->type == '(') {
			c = rnode_count(n->left);
			c += 2;
		}
	}
	if (!n->min && !n->max)
		return 0;
	if (n->min == 1 && n->max == 1)
		return c;
	if (n->max < 0)
		c = c * (n->min + 1) + 2;
	else
		c = c * (n->max + n->min) + n->max - n->min;
	return c + (n->min == 0);
}
