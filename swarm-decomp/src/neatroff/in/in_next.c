#include "vars.h"
#include "fn_in_next.h"
#include "fn_in_so.h"
#include "in_pop.h"
#include "libc.h"

int
in_next(void)
{
	int c;

	for (;;) {
		while (buf != NULL) {
			if (buf->nback != 0) {
				buf->nback--;
				return buf->back[buf->nback];
			}
			if (buf->name != NULL) {
				if (buf->pos < buf->len)
					return (unsigned char)buf->name[buf->pos++];
			} else {
				c = getc(buf->fp);
				if (c >= 0) {
					if (buf->last == '\n')
						buf->lnum++;
					buf->last = c;
					return c;
				}
			}
			in_pop();
		}
		if (cfile < nfiles)
			in_so(files[cfile++]);
		else
			return -1;
	}
}
