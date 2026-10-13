#include <stdlib.h>
#include "rnode.h"
#include "rnode_free.h"

static void rnode_free(struct rnode *n)
{
	if (n->left)
		rnode_free(n->left);
	if (n->right)
		rnode_free(n->right);
	free(n->str);
	free(n);
}
