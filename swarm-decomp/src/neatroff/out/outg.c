#include "c_ec.h"
#include "fn_out.h"
#include "fn_utf8one.h"

static void outg(char *s, int f, int sz) {
	int of = o_f;
	int os = o_s;

	if (f >= 0 && o_f != f) {
		o_f = f;
		out("f%d\n", f);
	}
	if (o_s != sz) {
		o_s = sz;
		out("s%d\n", sz);
	}
	if (utf8one(s)) {
		outnn("c%s%s", s, s[1] ? "\n" : "");
	} else {
		if (*s == c_ec) {
			if (s[1] == '(')
				s += 2;
		}
		out("C%s\n", s);
	}
	if (of >= 0 && of != o_f) {
		o_f = of;
		out("f%d\n", of);
	}
	if (o_s != os) {
		o_s = os;
		out("s%d\n", os);
	}
}
