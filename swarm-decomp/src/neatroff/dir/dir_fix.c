#include "fn_clr_get.h"
#include "fn_clr_str.h"
#include "fn_escread.h"
#include "fn_map.h"
#include "fn_nreg.h"
#include "fn_sbuf_append.h"
#include "c_ec.h"
#include "sbuf.h"

void dir_fix(struct sbuf *b, char *s)
{
	char *start;
	char *q;
	char *t;
	char *p;
	char tmp[32];
	char buf[1032];
	int size;
	int font;
	int color;
	int c;
	int n;
	long len;

	start = s;
	size = color = font = -1;
	p = s;
	dir_cd = *nreg(map(".td"));
	for (;;) {
		c = escread(&p, &t);
		if (c < 0)
			break;
		buf[0] = 0;
		switch (c) {
		case 0:
		case 'D':
		case 'h':
		case 'v':
		case 'x':
			len = p - start;
			memcpy(buf, start, len);
			buf[len] = 0;
			dir_append(buf);
			break;
		case '<':
			sprintf(tmp, "%cm[%s]", c_ec, clr_str(color));
			if (color >= 0)
				dir_append(tmp);
			sprintf(tmp, "%cf(%02d", c_ec, font);
			if (font >= 0)
				dir_append(tmp);
			if (size > 99) {
				sprintf(tmp, "%cs[%d]", c_ec, size);
				dir_append(tmp);
				dir_cd = 1;
				sprintf(tmp, "%cs[%d]", c_ec, size);
				dir_append(tmp);
			} else {
				sprintf(tmp, "%cs(%02d", c_ec, size);
				if (size >= 0) {
					dir_append(tmp);
					dir_cd = 1;
					sprintf(tmp, "%cs(%02d", c_ec, size);
					dir_append(tmp);
				} else {
					dir_cd = 1;
					sprintf(tmp, "%cs(%02d", c_ec, size);
				}
			}
			sprintf(tmp, "%cf(%02d", c_ec, font);
			if (font >= 0)
				dir_append(tmp);
			sprintf(tmp, "%cm[%s]", c_ec, clr_str(color));
			if (color >= 0)
				dir_append(tmp);
			break;
		case '>':
			sprintf(tmp, "%cm[%s]", c_ec, clr_str(color));
			if (color >= 0)
				dir_append(tmp);
			sprintf(tmp, "%cf(%02d", c_ec, font);
			if (font >= 0)
				dir_append(tmp);
			if (size > 99) {
				sprintf(tmp, "%cs[%d]", c_ec, size);
				dir_append(tmp);
				dir_cd = 0;
				sprintf(tmp, "%cs[%d]", c_ec, size);
				dir_append(tmp);
			} else {
				sprintf(tmp, "%cs(%02d", c_ec, size);
				if (size >= 0) {
					dir_append(tmp);
					dir_cd = 0;
					sprintf(tmp, "%cs(%02d", c_ec, size);
					dir_append(tmp);
				} else {
					dir_cd = 0;
					sprintf(tmp, "%cs(%02d", c_ec, size);
				}
			}
			sprintf(tmp, "%cf(%02d", c_ec, font);
			if (font >= 0)
				dir_append(tmp);
			sprintf(tmp, "%cm[%s]", c_ec, clr_str(color));
			if (color >= 0)
				dir_append(tmp);
			break;
		case 'X':
			sprintf(buf, "%c%c\003%s\003", c_ec, 'X', t);
			dir_append(buf);
			break;
		case 'f':
			n = strtol(t, 0, 10);
			if (n != font) {
				sprintf(tmp, "%cf(%02d", c_ec, font);
				if (font >= 0)
					dir_append(tmp);
				sprintf(tmp, "%cf(%02d", c_ec, n);
				if (n >= 0)
					dir_append(tmp);
			}
			font = n;
			break;
		case 'm':
			n = clr_get(t);
			if (n != color) {
				sprintf(tmp, "%cm[%s]", c_ec, clr_str(color));
				if (color >= 0)
					dir_append(tmp);
				sprintf(tmp, "%cm[%s]", c_ec, clr_str(n));
				if (n >= 0)
					dir_append(tmp);
			}
			color = n;
			break;
		case 's':
			n = strtol(t, 0, 10);
			if (n != size) {
				if (size > 99) {
					sprintf(tmp, "%cs[%d]", c_ec, size);
					dir_append(tmp);
				} else {
					sprintf(tmp, "%cs(%02d", c_ec, size);
					if (size >= 0)
						dir_append(tmp);
				}
				if (n > 99) {
					sprintf(tmp, "%cs[%d]", c_ec, n);
				} else {
					sprintf(tmp, "%cs(%02d", c_ec, n);
					if (n < 0)
						goto sdone;
				}
				dir_append(tmp);
			}
sdone:
			size = n;
			break;
		default:
			break;
		}
		start = p;
	}

	/* end of text: flush the state */
	sprintf(tmp, "%cm[%s]", c_ec, clr_str(color));
	if (color >= 0)
		dir_append(tmp);
	sprintf(tmp, "%cf(%02d", c_ec, font);
	if (font >= 0)
		dir_append(tmp);
	if (size > 99) {
		sprintf(tmp, "%cs[%d]", c_ec, size);
	} else {
		sprintf(tmp, "%cs(%02d", c_ec, size);
		if (size < 0)
			goto endf;
	}
	dir_append(tmp);
endf:
	dir_flush();
	if (*nreg(map(".td")) > 0) {
		q = dbuf + dbuf_sz - dbuf_n;
		q[dbuf_n] = 0;
	} else {
		q = dbuf;
		q[dbuf_n] = 0;
	}
	dbuf_n = 0;
	sbuf_append(b, q);
}
