#include <stdio.h>

static void box_bracketsel(int a, int b, int c, char **arr, int flag8, int flag9)
{
	int i;

	for (i = 0; arr[i]; i++) {
		printf(".if '%s'' ", sreg(a));
		printf(".if \\w'%s' ", arr[i]);
		if (flag9) {
			printf(".if (%s-(%dm/100)*2)<=(-\\n[bbury]+\\n[bblly]+(%dm/100*2)) ", nreg(b), e_rulethickness, e_axisheight);
			printf(".if (%s*2)<=(-\\n[bbury]+\\n[bblly]-(%dm/100*2)) ", nreg(c), e_axisheight);
		} else
			printf(".if (%s+%s)<=(-\\n[bbury]+\\n[bblly]) ", nreg(b), nreg(c));
		printf(".ds %s \"%s\n", sregname(a), arr[i]);
	}
	if (flag8)
		while (--i >= 0)
			printf(".if '%s'' .if \\w'%s' .ds %s \"%s\n", sreg(a), arr[i], sregname(a), arr[i]);
}
