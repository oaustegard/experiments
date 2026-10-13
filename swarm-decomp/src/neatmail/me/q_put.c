static void q_put(int c)
{
	unsigned v, r;
	int i, n;

	n = out_mlen + 1;
	v = (out_mbuf << 8) | c;
	out_mlen = n;
	out_mbuf = v;
	if (n == 3) {
		i = out_n;
		r = (unsigned char)"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[v & 0x3f];
		r = (r << 8) | (unsigned char)"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[(v >> 6) & 0x3f];
		out_mlen = 0;
		out_mbuf = 0;
		r = (r << 8) | (unsigned char)"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[(v >> 12) & 0x3f];
		r = (r << 8) | (unsigned char)"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[v >> 18 & 0x3f];
		*(unsigned *)(out + i) = r;
		out_n = i + 4;
	}
}
