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
	char *x;
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
		printf(".nr %s %s-(%s+%s+%s/2)*11/10/%s\n", nregname(n2), nreg(nid),
		    nreg(d[1]), nreg(a[1]), nreg(c[1]), nreg(b[1]));
		printf(".if %s<0 .nr %s 0\n", nreg(n2), nregname(n2));
		printf(".nr %s 0%s*2\n", nregname(n0), nreg(n2));
		printf(".de %s\n", sregname(s));
		printf(".if \\%s=\\%s .as %s \"\\v'-\\%su'%s\\h'-\\%su'\\v'-\\%su'\n",
		    nreg(n1), nreg(n2), sregname(sid), nreg(d[3]), p3, nreg(d[0]),
		    nreg(d[2]));
	} else {
		printf(".nr %s %s*2-%s-%s*11/10/%s\n", nregname(n0), nreg(nid),
		    nreg(a[1]), nreg(c[1]), nreg(b[1]));
		printf(".if %s<0 .nr %s 0\n", nreg(n0), nregname(n0));
		printf(".de %s\n", sregname(s));
	}
	printf(".if \\%s<\\%s .as %s \"\\v'-\\%su'%s\\h'-\\%su'\\v'-\\%su'\n",
	    nreg(n1), nreg(n0), sregname(sid), nreg(b[3]), p1, nreg(b[0]),
	    nreg(b[2]));
	printf(".if \\\\n+%s<\\%s .%s\n", escarg(nregname(n1)), nreg(n0),
	    sregname(s));
	puts("..");
	printf(".ds %s \"\\v'-%su'%s\\h'-%su'\\v'-%su'\n", sregname(sid),
	    nreg(c[3]), p2, nreg(c[0]), nreg(c[2]));
	printf(".nr %s 0 1\n", nregname(n1));
	printf(".%s\n", sregname(s));
	printf(".as %s \"\\v'-%su'%s\\h'-%su'\\v'-%su'\n", sregname(sid),
	    nreg(a[3]), p0, nreg(a[0]), nreg(a[2]));
	if (p3) {
		printf(".as %s \"\\v'%su*%su+%su+%su+%su'\n", sregname(sid),
		    nreg(n0), nreg(b[1]), nreg(c[1]), nreg(a[1]), nreg(d[1]));
		x = nreg(d[1]);
	} else {
		printf(".as %s \"\\v'%su*%su+%su+%su+%su'\n", sregname(sid),
		    nreg(n0), nreg(b[1]), nreg(c[1]), nreg(a[1]), "0");
		x = nreg(b[0]);
	}
	printf(".as %s \"\\h'%su'\n", sregname(sid), x);
	for (i = 0; i < 4; i++)
		nregrm(a[i]);
	for (i = 0; i < 4; i++)
		nregrm(b[i]);
	for (i = 0; i < 4; i++)
		nregrm(c[i]);
	if (p3)
		for (i = 0; i < 4; i++)
			nregrm(d[i]);
	nregrm(n0);
	nregrm(n1);
	nregrm(n2);
	sregrm(s);
}
