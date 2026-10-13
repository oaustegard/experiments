#include "CloseP.h"
#include "fn_luaF_close.h"

static void closepaux (lua_State *L, void *ud) {
  struct CloseP *pcl = (struct CloseP *)ud;
  luaF_close(L, pcl->level, pcl->status, 0);
}
