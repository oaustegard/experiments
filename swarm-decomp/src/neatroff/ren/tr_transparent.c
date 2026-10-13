void tr_transparent(struct tr_args *a)
{
	if (cdiv)
		sbuf_printf(&cdiv->buf, "%s\n", a->arg[0]);
	else
		out("%s\n", a->arg[0]);
}
