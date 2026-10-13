int eval(char *s, int u)
{
	char *p = s;

	defunit = u;
	if (u == 'v') {
		abspos = -*nreg('d') - wb_vpos(env_wb());
	} else if (u == 'm') {
		abspos = *nreg(map(".b0")) - f_hpos();
	}
	return evalexpr(&p);
}
