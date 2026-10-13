#include "box.h"
#include "fn_nreg.h"
#include "fn_sreg.h"
#include "fn_escarg.h"
#include "e_axisheight.h"
#include "box_bracketmk.h"

static void box_bracketmk(int sid, int nid, char *p0, char *p1, char *p2, char *p3)
{
	int n0 = nregmk();
	int n1 = nregmk();
	int n2 = nregmk();
	int s = sregmk();
	int a[4], b[4], c[4], d[4];
	int i;

	for (i = 0; i < 4; i++)
		a[i] = nregmk();
	tok_len(p0, a[0], a[1], a[2], a[3]);
	for (i = 0; i < 4; i++)
		b[i] = nregmk();
	tok_len(p1, b[0], b[1], b[2], b[3]);
	for (i = 0; i < 4; i++)
		c[i] = nregmk();
	tok_len(p2, c[0], c[1], c[2], c[3]);
	if (p3) {
		for (i = 0; i < 4; i++)
			d[i] = nregmk();
		tok_len(p3, d[0], d[1], d[2], d[3]);
		printf(".nr %s %s-(%s+%s+%s/2)*11/10/%s\n", nregname(n2), nreg(c[1]),
		    nreg(a[1]), nreg(d[1]), nreg(nid), nregname(n2));
		printf(".if %s<0 .nr %s 0\n", nreg(n2), nregname(n2));
		printf(".nr %s 0%s*2\n", nregname(n1), nreg(nid));
		printf(".de %s\n", sregname(s));
		printf(".as %s \"\\v'-%su'%s\\h'-%su'\\v'-%su'\n", sregname(s), nreg(c[0]),
		    sregname(sid), nreg(d[0]), nreg(b[0]));
		puts("..");
		printf(".if \\\\n+%s<\\%s .%s\n", escarg(nregname(n0)), nreg(n1), sregname(sid));
		printf(".ds %s \"\\v'-%su'%s\\h'-%su'\\v'-%su'\n", sregname(sid), nreg(b[1]),
		    sregname(s), nreg(c[1]), nreg(d[1]));
		nregname(n0);
	} else {
		printf(".nr %s %s*2-%s-%s*11/10/%s\n", nregname(n2), nreg(c[1]), nreg(a[1]),
		    nreg(b[1]), nregname(n0));
		printf(".if %s<0 .nr %s 0\n", nregname(n0), nreg(n0));
		printf(".de %s\n", sregname(sid));
		puts("..");
	}
	printf(".%s\n", sregname(sid));
	printf(".as %s \"\\v'-%su'%s\\h'-%su'\\v'-%su'\n", sregname(sid), nreg(b[2]),
	    sregname(s), nreg(c[2]), nreg(d[2]));
	for (i = 0; i < 4; i++)
		nregrm(a[i]);
	for (i = 0; i < 4; i++)
		nregrm(b[i]);
	for (i = 0; i < 4; i++)
		nregrm(c[i]);
	for (i = 0; i < 4; i++)
		nregrm(d[i]);
	nregrm(n0);
	nregrm(n1);
	nregrm(n2);
	sregrm(s);
}
