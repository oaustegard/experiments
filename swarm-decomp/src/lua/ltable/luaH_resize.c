#include "lstate.h"
#include "Table.h"

void luaH_resize (lua_State *L, Table *t, unsigned int newasize, unsigned int nhsize) {
  unsigned int i;
  Table newt;  /* to keep the new hash part */
  unsigned int oldasize = setlimittosize(t);
  TValue *newarray;
  /* create new hash part with appropriate size into 'newt' */
  setnodevector(L, &newt, nhsize);
  if (newasize < oldasize) {  /* will array shrink? */
    t->alimit = newasize;  /* pretend array has new size... */
    exchangehashpart(t, &newt);  /* and new hash */
    /* re-insert into the new hash the elements from vanishing slice */
    for (i = newasize; i < oldasize; i++) {
      if (t->array[i].tt_ & 0xf) {
        lua_Integer key = i + 1;
        const TValue *p = luaH_getint(t, key);
        if (p->tt_ == 0x20) {
          TValue k;
          k.value_.i = key;
          k.tt_ = 0x03;
          luaH_newkey(L, t, &k, &t->array[i]);
        }
        else {
          TValue *q = (TValue *)p;
          q->value_ = t->array[i].value_;
          q->tt_ = t->array[i].tt_;
        }
      }
    }
    t->alimit = oldasize;  /* restore current size... */
    exchangehashpart(t, &newt);  /* and hash (in case of errors) */
  }
  /* allocate new array */
  newarray = (TValue *)luaM_realloc_(L, t->array, oldasize * sizeof(TValue), newasize * sizeof(TValue));
  if (luai_unlikely(newarray == NULL && newasize > 0)) {  /* allocation failed? */
    freehash(L, &newt);  /* release new hash part */
    luaD_throw(L, 4);  /* raise error (with array unchanged) */
  }
  /* allocation ok; initialize new part of the array */
  exchangehashpart(t, &newt);  /* 't' has the new hash ('newt' has the old) */
  t->array = newarray;  /* set new array part */
  t->alimit = newasize;
  for (i = oldasize; i < newasize; i++)  /* clear new slice of the array */
    t->array[i].tt_ = 0x10;
  reinsert(L, &newt, t);  /* 'newt' now has the old hash */
  freehash(L, &newt);  /* free old hash part */
}
