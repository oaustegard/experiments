#pragma once

/* one message of the mk index: 88 (0x58) bytes, from the mk listing
 * (msgcmp_id, msgcmp_date, put_msg and the mk main loop) */
struct mkmsg {
	char *s;		/* 0x00: message start (from mbox_get) */
	long n;			/* 0x08: message length */
	char *id;		/* 0x10: Message-ID value, inside <> */
	int idlen;		/* 0x18 */
	char *irt;		/* 0x20: In-Reply-To value, inside <> */
	int irtlen;		/* 0x28 */
	int date;		/* 0x2c: datedate() of the Date: header */
	int depth;		/* 0x30: thread depth */
	int idx;		/* 0x34: message index in the mailbox */
	struct mkmsg *parent;	/* 0x38 */
	struct mkmsg *child;	/* 0x40: first child */
	struct mkmsg *lastchild;	/* 0x48 */
	struct mkmsg *next;	/* 0x50: next sibling */
};
