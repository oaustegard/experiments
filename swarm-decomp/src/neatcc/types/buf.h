#pragma once
#include "macro.h"
/* one entry of the cpp input-buffer stack; 0x80b0 bytes */
struct buf {
	char *buf;		/* +0 */
	long len;		/* +8 */
	long cur;		/* +16 */
	int kind;		/* +24 */
	char rest[160 - 28];	/* +28: file name of an include */
	struct macro *macro;	/* +160: macro being expanded */
	char arg[32][1024];	/* +168: argument values of that macro */
	int argi;		/* +32936: argument index */
	char pad1[4];
};
