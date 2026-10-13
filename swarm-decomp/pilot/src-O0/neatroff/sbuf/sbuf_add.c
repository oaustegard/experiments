void sbuf_add(struct sbuf *sb, int c)
{
	if (sb->n + 2 >= sb->sz)
		sbuf_extend(sb, sb->sz * 2);
	sb->s[sb->n++] = c;
}
