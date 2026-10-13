#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include "rnode.h"
#include "rnode_free.h"
#include "rnode_parse.h"
#include "rnode_seq.h"

static struct rnode *rnode_seq(char **s)
{
	char *p = *s;
	char *q, *e;
	struct rnode *n = NULL, *m, *r;
	unsigned char c;
	int acc, len;

	c = *p;
	if (c == 0)
		goto out;
	if (c == '|' || c == ')')
		goto out;
	if (c != '(') {
		n = calloc(sizeof(struct rnode), 1);
		n->min = 1;
		n->max = 1;
		switch (*p) {
		case '[':
			len = brk_len(p);
			n->c00 = '[';
			goto copy;
		case '$':
			p++;
			n->c00 = '$';
			*s = p;
			break;
		case '.':
			p++;
			n->c00 = '.';
			*s = p;
			break;
		case '\\':
			if ((p[1] & ~2) == '<') {
				n->c00 = p[1] == '<' ? '<' : '>';
				p += 2;
				*s = p;
				break;
			}
			p++;
			*s = p;
			goto literal;
		case '^':
			p++;
			n->c00 = '^';
			*s = p;
			break;
		default:
			goto literal;
		}
		goto quant;
	literal:
		q = p;
		do {
			if (q == p) {
				q = p + uc_len(p);
				continue;
			}
			if (strchr(".^$[(|)*?+{\\", *q))
				break;
			e = q + uc_len(q);
			if (*e && strchr("*?+{", *e))
				break;
			q = e;
		} while (q != p);
		len = q - p;
	copy:
		n->str = malloc(len + 1);
		memcpy(n->str, p, len);
		p += len;
		*s = p;
		n->str[len] = 0;
	} else {
		*s = p + 1;
		if (p[1] == ')') {
			r = NULL;
		} else {
			r = rnode_parse(s);
			if (!r)
				goto out;
			if (**s != ')') {
				rnode_free(r);
				goto out;
			}
		}
		p = *s + 1;
		*s = p;
		n = calloc(sizeof(struct rnode), 1);
		n->left = r;
		n->type = '(';
		n->min = 1;
		n->max = 1;
	}
quant:
	c = *p;
	if (c == '*' || c == '?') {
		n->min = 0;
		n->max = (*p != '*') * 2 - 1;
		p++;
		*s = p;
		c = *p;
	}
	if (c == '+') {
		n->min = 1;
		n->max = -1;
		p++;
		*s = p;
		c = *p;
	}
	if (c == '{') {
		n->min = 0;
		n->max = 0;
		e = p + 1;
		*s = e;
		if (isdigit(*e)) {
			acc = 0;
			do {
				acc = acc * 10 + (*e - '0');
				n->min = acc;
				*s = ++e;
			} while (isdigit(*e));
		}
		if (*e == ',') {
			e++;
			*s = e;
			if (*e == '}')
				n->max = -1;
			if (isdigit(*e)) {
				do {
					n->max = n->max * 10 + (*e - '0');
					*s = ++e;
				} while (isdigit(*e));
			}
		} else {
			n->max = n->min;
		}
		*s = e + 1;
		goto concat;
	}
concat:
	m = rnode_seq(s);
	if (m) {
		r = calloc(sizeof(struct rnode), 1);
		r->left = n;
		r->right = m;
		r->type = 'c';
		r->min = 1;
		r->max = 1;
		n = r;
	}
out:
	return n;
}
