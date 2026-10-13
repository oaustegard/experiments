static void sbuf_extend(struct sbuf *sb, int n)
{
	sb->sz = (n + 0x1ff) & ~0x1ff;
	sb->s = mextend(sb->s, sb->n, sb->sz, 1);
}
