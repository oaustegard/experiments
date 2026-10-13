#include "cpp_protos.h"

void cpp_define(char *name, char *val)
{
	char b[2048];
	sprintf(b, "%s\t%s", name, val);
	buf_new(4, b, strlen(b));
	macro_define();
	buf_pop();
}
