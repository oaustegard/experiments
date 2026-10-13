#include "libc.h"
#include "fn_tok_jmp.h"
#include "fn_tok_poptext.h"
#include "fn_tok_delim.h"
#include "fn_tok_macro.h"
#include "fn_escarg.h"
#include "fn_def_set.h"
#include "fn_def_brcostput.h"
#include "fn_def_piecesput.h"
#include "fn_def_sizesput.h"
#include "fn_def_typeput.h"

static int eqn_commands(void)
{
	char *s;
	int v;

	if (tok_jmp("delim") == 0) {
		tok_delim();
		return 0;
	}
	if (tok_jmp("define") == 0) {
		tok_macro();
		return 0;
	}
	if (tok_jmp("gfont") == 0) {
		s = tok_poptext(1);
		if (s == NULL)
			s = "";
		else if (*s == '"') {
			s[strlen(s) - 1] = '\0';
			s++;
		}
		strcpy(gfont, s);
		return 0;
	}
	if (tok_jmp("grfont") == 0) {
		s = tok_poptext(1);
		if (s == NULL)
			s = "";
		else if (*s == '"') {
			s[strlen(s) - 1] = '\0';
			s++;
		}
		strcpy(grfont, s);
		return 0;
	}
	if (tok_jmp("gbfont") == 0) {
		s = tok_poptext(1);
		if (s == NULL)
			s = "";
		else if (*s == '"') {
			s[strlen(s) - 1] = '\0';
			s++;
		}
		strcpy(gbfont, s);
		return 0;
	}
	if (tok_jmp("gsize") == 0) {
		s = tok_poptext(1);
		if (s == NULL)
			s = "";
		else {
			if (*s == '"') {
				s[strlen(s) - 1] = '\0';
				s++;
			}
			if (*s == '+' || *s == '-') {
				sprintf(gsize, "\\n%s%s", escarg(".eqnsz"), s);
				return 0;
			}
		}
		strcpy(gsize, s);
		return 0;
	}
	if (tok_jmp("set") == 0) {
		char name[32];

		strcpy(name, tok_poptext(1));
		v = strtol(tok_poptext(1), NULL, 10);
		def_set(name, v);
		return 0;
	}
	if (tok_jmp("bracketsizes") == 0) {
		char name[64];
		char strs[8][64];
		const char *sizes[8] = {0};
		int i, n;

		s = tok_poptext(1);
		if (s == NULL)
			s = "";
		else if (*s == '"') {
			s[strlen(s) - 1] = '\0';
			s++;
		}
		snprintf(name, 64, "%s", s);
		n = strtol(tok_poptext(1), NULL, 10);
		for (i = 0; i < n; i++) {
			s = tok_poptext(1);
			if (s == NULL)
				s = "";
			else if (*s == '"') {
				s[strlen(s) - 1] = '\0';
				s++;
			}
			if (i <= 7) {
				snprintf(strs[i], 64, "%s", s);
				sizes[i] = strs[i];
			}
		}
		def_sizesput(name, sizes);
		return 0;
	}
	if (tok_jmp("bracketpieces") == 0) {
		char a[64], b[64], c[64], d[64], e[64];

		s = tok_poptext(1);
		if (s == NULL)
			s = "";
		else if (*s == '"') {
			s[strlen(s) - 1] = '\0';
			s++;
		}
		snprintf(a, 64, "%s", s);
		s = tok_poptext(1);
		if (s == NULL)
			s = "";
		else if (*s == '"') {
			s[strlen(s) - 1] = '\0';
			s++;
		}
		snprintf(b, 64, "%s", s);
		s = tok_poptext(1);
		if (s == NULL)
			s = "";
		else if (*s == '"') {
			s[strlen(s) - 1] = '\0';
			s++;
		}
		snprintf(c, 64, "%s", s);
		s = tok_poptext(1);
		if (s == NULL)
			s = "";
		else if (*s == '"') {
			s[strlen(s) - 1] = '\0';
			s++;
		}
		snprintf(d, 64, "%s", s);
		s = tok_poptext(1);
		if (s == NULL)
			s = "";
		else if (*s == '"') {
			s[strlen(s) - 1] = '\0';
			s++;
		}
		snprintf(e, 64, "%s", s);
		def_piecesput(a, b, c, d, e);
		return 0;
	}
	if (tok_jmp("chartype") == 0) {
		char a[32], b[32];

		s = tok_poptext(1);
		if (s == NULL)
			s = "";
		else if (*s == '"') {
			s[strlen(s) - 1] = '\0';
			s++;
		}
		snprintf(a, 32, "%s", s);
		s = tok_poptext(1);
		if (s == NULL)
			s = "";
		else if (*s == '"') {
			s[strlen(s) - 1] = '\0';
			s++;
		}
		snprintf(b, 32, "%s", s);
		if (strcmp(a, "ord") == 0 || strcmp(a, "ordinary") == 0)
			v = 0x10;
		else {
			v = typenum(a);
			if (v < 0)
				return 0;
		}
		def_typeput(b, v);
		return 0;
	}
	if (tok_jmp("breakcost") != 0)
		return 1;
	else {
		char a[32];
		int cost, t;

		s = tok_poptext(1);
		if (s == NULL)
			s = "";
		else if (*s == '"') {
			s[strlen(s) - 1] = '\0';
			s++;
		}
		snprintf(a, 32, "%s", s);
		cost = strtol(tok_poptext(1), NULL, 10);
		if (strcmp(a, "any") == 0)
			t = 0;
		else if (strcmp(a, "ord") == 0 || strcmp(a, "ordinary") == 0)
			t = 0x10;
		else {
			t = typenum(a);
			if (t < 0)
				return 0;
		}
		def_brcostput(t, cost);
	}
	return 0;
}
