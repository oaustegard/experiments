void tok_eqnout(char *s)
{
	if (tok_part) {
		printf(".as %s \"%s%s%s\n", "10", "\\E*[.eqnbeg]\\R'.eqnfn0 \\En(.f'\\R'.eqnsz0 \\En(.s'", s, "\\f[\\En[.eqnfn0]]\\s[\\En[.eqnsz0]]\\E*[.eqnend]");
		return;
	}
	printf(".ds %s \"%s%s%s\n", "10", "\\E*[.eqnbeg]\\R'.eqnfn0 \\En(.f'\\R'.eqnsz0 \\En(.s'", s, "\\f[\\En[.eqnfn0]]\\s[\\En[.eqnsz0]]\\E*[.eqnend]");
	printf(".lf %d\n", src_lineget() - 1);
	printf("\\&\\*%s\n", escarg("10"));
}
