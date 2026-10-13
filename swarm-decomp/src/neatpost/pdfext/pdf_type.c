#include <ctype.h>
#include "sys_std.h"

int pdf_type(unsigned char *s, int end, int i)
{
	int pos = i;
	int c;

	while (pos < end && isspace(s[pos]))
		pos++;
	c = s[pos];
	if (c == '/')
		return '/';
	if (c == '(')
		return 's';
	if (c == '<')
		return s[pos + 1] != '<' ? 's' : 'd';
	if (c == '[')
		return 'l';
	if (strchr("0123456789+-.", c) == 0)
		return -1;
	if (!isdigit(c))
		return 'n';
	while (pos < end && isdigit(s[pos]))
		pos++;
	if (pos >= end)
		return 'n';
	while (pos < end && isspace(s[pos]))
		pos++;
	if (!isdigit(s[pos]) || pos >= end)
		return 'n';
	while (pos < end && isdigit(s[pos]))
		pos++;
	if (pos >= end)
		return 'n';
	while (pos < end && isspace(s[pos]))
		pos++;
	if (pos >= end)
		return 'n';
	return s[pos] == 'R' ? 'r' : 'n';
}
