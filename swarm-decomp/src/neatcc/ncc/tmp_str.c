static char *tmp_str(char *s, int n)
{
	static char name[128];
	static int id;

	sprintf(name, "__neatcc.s%d", id++);
	s[n++] = 0;
	o_dscpy(o_dsnew(name, n, 0), s, n);
	return name;
}
