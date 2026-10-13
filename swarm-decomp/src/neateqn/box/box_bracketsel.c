static void box_bracketsel(int a, int b, int c, char **s, int d, int e)
{
	int i = -1;

	if (s[0]) {
		i = 0;
		do {
			printf(".if '%s'' ", sreg(a));
			printf(".if \\w'%s' ", s[i]);
			if (e) {
				printf(".if (%s-(%dm/100)*2)<=(-\\n[bbury]+\\n[bblly]+(%dm/100*2)) ", nreg(b), e_rulethickness, e_axisheight);
				printf(".if (%s*2)<=(-\\n[bbury]+\\n[bblly]-(%dm/100*2)) ", nreg(c), e_axisheight);
			} else {
				printf(".if (%s+%s)<=(-\\n[bbury]+\\n[bblly]) ", nreg(b), nreg(c));
			}
			printf(".ds %s \"%s\n", sregname(a), s[i]);
			i++;
		} while (s[i]);
		i--;
	}
	if (d) {
		if (i >= 0)
			for (; i >= 0; i--)
				printf(".if '%s'' .if \\w'%s' .ds %s \"%s\n", sreg(a), s[i], sregname(a), s[i]);
	}
}
