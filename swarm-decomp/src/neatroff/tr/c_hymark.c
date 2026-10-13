#include "ext.h"

int c_hymark(char *s)
{
	if (*s == 0x5c)
		return (s[1] == 0x3a && s[2] == 0) || strcmp(env_hc(), s) == 0;
	return strcmp(env_hc(), s) == 0;
}
