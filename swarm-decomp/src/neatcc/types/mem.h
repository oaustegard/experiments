#pragma once
/* growable byte buffer: data pointer, allocated size, used length */
struct mem {
	char *buf;	/* +0 */
	long cap;	/* +8 */
	long len;	/* +16 */
};
