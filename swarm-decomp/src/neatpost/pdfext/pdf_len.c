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
	}
	if (c == '<') {
		if (end > pos + 1 && s[pos + 1] == '<') {
			pos += 2;
			while (pos + 2 < end) {
				if (s[pos] == '>' && s[pos + 1] == '>') {
					if (pos + 2 < end)
						pos += 2;
					break;
				}
				pos += pdf_len(s, end, pos);
				if (pos < end)
					pos += pdf_len(s, end, pos);
				while (pos < end && isspace(s[pos]))
					pos++;
			}
		} else {
			while (pos < end && s[pos] != '>')
				pos++;
			if (pos < end)
				pos++;
		}
	} else if (c == '[') {
		pos++;
		while (pos < end && s[pos] != ']') {
			pos += pdf_len(s, end, pos);
			while (pos < end && isspace(s[pos]))
				pos++;
		}
		pos++;
	} else if (c == '(') {
		pos++;
		depth = 1;
		while (pos < end && depth > 0) {
			if (s[pos] == '(')
				depth++;
			else if (s[pos] == ')')
				depth--;
			else if (s[pos] == '\\')
				pos++;
			pos++;
		}
	} else if (c == '/') {
		pos++;
		while (pos < end && !strchr(" \t\r\n\f()<>[]{}/%", s[pos]))
			pos++;
	}
	return pos - i;
}
