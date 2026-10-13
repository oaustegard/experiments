#include <string.h>

#include "fn_charnext.h"

int charnext_delim(char *buf, int (*next)(void), void (*back)(void), char *delim)
{
	int c = charnext(buf, next, back);
	return strcmp(buf, delim) == 0 ? -1 : c;
}
