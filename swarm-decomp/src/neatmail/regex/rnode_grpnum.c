#include "rnode.h"
#include "rnode_grpnum.h"

static int rnode_grpnum(struct rnode *n, int g)
{
	int c = 0;

	for (; n; n = n->right) {
		int s = 0;
		if (n->type == '(') {
			n->c28 = g;
			s = 1;
		}
		s += rnode_grpnum(n->left, g + s);
		c += s;
		g += s;
	}
	return c;
}
