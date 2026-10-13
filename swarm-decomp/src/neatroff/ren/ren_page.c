#include "trap_exec.h"
#include "trap_first.h"
#include "ren_page.h"

static void ren_page(int n)
{
	int b;
	int i;

	if (n == 0 && bp_final > 1)
		return;
	*nreg(map("nl")) = 0;
	*nreg('d') = 0;
	*nreg('h') = 0;
	b = bp_next;
	if (b == 0x40000000)
		b = *nreg(map("%")) + 1;
	{
		int *p = nreg(map("%"));

		bp_next = 0x40000000;
		*p = b;
	}
	++*nreg('%');
	out("p%d\n", *nreg('%'));
	out("V%d\n", 0);
	if (trap_pos(-1))
		return;
	i = trap_first(-1);
	if (cdiv) {
		int t = -1;

		if (cdiv->f18 && cdiv->f14 >= 0)
			t = cdiv->f18;
		trap_exec(t);
		return;
	}
	if (i < 0) {
		trap_exec(-1);
		return;
	}
	trap_exec(treg[i]);
}
