int f_hpos(void)
{
	int w = fmt_wid(env_fmt());
	return w + wb_wid(env_wb());
}
