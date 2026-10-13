#include <ctype.h>
#include "sys_std.h"
#include "fn_pdf_type.h"

int pdf_len(unsigned char *s, int end, int i)
{
	int pos = i;
	int c;
	int depth;

	if (i >= end)
		return 0;
	while (pos < end && isspace(s[pos]))
		pos++;
	c = s[pos];
	if (strchr("0123456789+-.", c)) {
		if (pdf_type(s, end, pos) == 'r')
			return (unsigned char *)memchr(s + pos, 'R', end - pos) - (s + i) + 1;
		pos++;
		while (pos < end && strchr("0123456789.", s[pos]))
			pos++;
		return pos - i;
	}
	if (c == '(') {
		pos++;
		if (end > pos) {
			depth = 1;
			do {
				c = s[pos];
				if (c == '(')
					depth++;
				else if (c == ')')
					depth--;
				else if (c == '\\')
					pos++;
				pos++;
			} while (depth > 0 && pos < end);
		}
		return pos - i;
	}
	if (c == '<') {
		if (end > pos + 1 && s[pos + 1] == '<') {
			pos += 2;
			while (pos < end && !(s[pos] == '>' && pos + 1 < end && s[pos + 1] == '>'))
				pos++;
			if (pos < end)
				pos += 2;
			return pos - i;
		}
		while (pos < end && s[pos] != '>')
			pos++;
		if (pos < end)
			pos++;
		return pos - i;
	}
	if (c == '/') {
		pos++;
		while (pos < end && !strchr(" \t\r\n\f()<>[]{}/%", s[pos]))
			pos++;
		return pos - i;
	}
	if (c == '[') {
		pos++;
		while (pos < end && s[pos] != ']')
			pos += pdf_len(s, end, pos);
		if (pos < end)
			pos++;
		return pos - i;
	}
	return pos - i;
}
