#include "lstate.h"
#include "Table.h"

unsigned int luaH_realasize (const Table *t);
void luaH_resize (lua_State *L, Table *t, unsigned int nasize, unsigned int nhsize);

static unsigned int setlimittosize (Table *t);
static unsigned int numusearray (const Table *t, unsigned int *nums);
static int numusehash (const Table *t, unsigned int *nums, unsigned int *pna);
static unsigned int computesizes (unsigned int nums[], unsigned int *pna);
static int countint (lua_Integer key, unsigned int *nums);

static void rehash (lua_State *L, Table *t, const TValue *ek) {
  unsigned int asize;
  unsigned int na;
  unsigned int nums[32];
  int i;
  int totaluse;
  for (i = 0; i <= 31; i++) nums[i] = 0;
  setlimittosize(t);
  na = numusearray(t, nums);
  totaluse = na;
  totaluse += numusehash(t, nums, &na);
  if (ek->tt_ == 0x03)
    na += countint(ek->value_.i, nums);
  totaluse++;
  asize = computesizes(nums, &na);
  luaH_resize(L, t, asize, totaluse - na);
}
