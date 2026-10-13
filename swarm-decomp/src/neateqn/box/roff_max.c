static void roff_max(int a, int b, int c)
{
	printf(".ie %s>=%s .nr %s 0+%s\n", nreg(b), nreg(c), nregname(a), nreg(b));
	printf(".el .nr %s 0+%s\n", nregname(a), nreg(c));
}
