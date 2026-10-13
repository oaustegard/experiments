#include "mbox.h"
#include "fn_mbox_savetag.h"

int mbox_save(struct mbox *m)
{
	return mbox_savetag(m, 0);
}
