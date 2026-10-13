#pragma once
/* one entry of the int-code table: op word, three operands, and a pointer
   freed when the op carries 0x200; 40 bytes */
struct ic_ent {
	long op;	/* +0 */
	long a;		/* +8 */
	long b;		/* +16 */
	long c;		/* +24 */
	char *d;	/* +32 */
};
