static void q_put(int c)
{
	unsigned v, r, a, b, e, d;
	int i, n;

	n = out_mlen + 1;
	v = (out_mbuf << 8) | c;
	out_mlen = n;
	out_mbuf = v;
	if (n == 3) {
		d = (unsigned char)"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[v >> 18 & 0x3f];
		i = out_n;
		out_mbuf = 0;
		b = (unsigned char)"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[v >> 6 & 0x3f];
		e = (unsigned char)"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[(v >> 12) % 64];
		a = (unsigned char)"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[v % 64];
		out_mlen = 0;
		r = (a << 8) | b;
		r = (r << 8) | e;
		r = (r << 8) | d;
		*(unsigned *)(out + i) = r;
		out_n = i + 4;
	}
}
