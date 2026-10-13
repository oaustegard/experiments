#include "libc.h"
#include "box.h"
#include "e_vars.h"
#include "fn_def_choppedset.h"
#include "def_macros.h"
#include "fn_src_define.h"
#include "fn_src_done.h"
#include "fn_src_lineget.h"
#include "fn_tok_eqn.h"
#include "fn_tok_get.h"
#include "fn_tok_jmp.h"
#include "fn_tok_inline.h"
#include "fn_tok_pop.h"
#include "fn_tok_eqnout.h"
#include "fn_tok_poptext.h"
#include "fn_reg_reset.h"
#include "fn_nreg.h"
#include "fn_escarg.h"
#include "fn_box_alloc.h"
#include "fn_box_free.h"
#include "fn_box_markpos.h"
#include "fn_box_merge.h"
#include "fn_box_vertspace.h"
#include "fn_box_empty.h"
#include "fn_box_toreg.h"
#include "fn_box_puttext.h"
#include "eqn_box.h"

int main(int argc, char **argv)
{
	char buf[128];
	struct box *b, *x;
	int i, col, ps;

	i = 1;
	if (i < argc) do {
		char *a = argv[i];
		char *arg;
		if (a[0] != '-' || a[1] == '\0')
			break;
		if (a[1] != 'c') {
			fwrite("Usage: neateqn [options] <input >output\n\nOptions:\n  -c chars  \tcharacters that chop equations\n", 1, 94, stderr);
			return 1;
		}
		if (a[2] != '\0') {
			arg = a + 2;
		} else {
			arg = argv[++i];
		}
		def_choppedset(arg);
		i++;
	} while (i < argc);
	{
		const char *name;
		i = 0;
		name = def_macros[i].name;
		while (name != NULL) {
			src_define((char *) name, (char *) def_macros[i].val);
			i++;
			name = def_macros[i].name;
		}
	}
	if (tok_eqn() == 0) {
		do {
			reg_reset();
			eqn_mk = 0;
			tok_pop();
			printf(".nr %s \\n(.s\n", ".eqnsz");
			printf(".nr %s \\n(.f\n", ".eqnfn");
			eqn_lineupreg = nregmk();
			col = 2 * (tok_inline() != 0);
			ps = nregmk();
			printf(".nr %s %s\n", nregname(ps), gsize);
			b = box_alloc(ps, 0, col);
			while (tok_get() != NULL) {
				if (tok_jmp("mark") == 0) {
					eqn_mk = eqn_mk == 0 ? 1 : eqn_mk;
					box_markpos(b, ".eqnmk");
				} else if (tok_jmp("lineup") != 0) {
					x = eqn_box(col, b, ps, NULL);
					box_merge(b, x, 1);
					box_free(x);
				} else {
					char *r;
					eqn_mk = 2;
					box_markpos(b, nregname(eqn_lineupreg));
					r = nreg(eqn_lineupreg);
					sprintf(eqn_lineup, "\\h'\\n%su-%su'", escarg(".eqnmk"), r);
				}
			}
			box_vertspace(b);
			nregrm(ps);
			printf(".nr MK %d\n", eqn_mk);
			if (box_empty(b) == 0) {
				sprintf(buf, "%s%s", eqn_lineup, box_toreg(b));
				tok_eqnout(buf);
				printf(".ps \\n%s\n", escarg(".eqnsz"));
				printf(".ft \\n%s\n", escarg(".eqnfn"));
			}
			printf(".lf %d\n", src_lineget());
			eqn_lineup[0] = '\0';
			nregrm(eqn_lineupreg);
			box_free(b);
		} while (tok_eqn() == 0);
	}
	src_done();
	return 0;
}
