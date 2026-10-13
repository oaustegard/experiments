#include <string.h>

void sbuf_init(struct sbuf *sb)
{
	memset(sb, 0, sizeof(*sb));
	sbuf_extend(sb, 0x200);
}
