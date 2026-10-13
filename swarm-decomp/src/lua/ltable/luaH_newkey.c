#include "lstate.h"
#include "Table.h"

int luaV_flttointeger (lua_Number n, lua_Integer *p, int mode);
void luaC_barrierback_ (lua_State *L, GCObject *o);
void luaH_resize (lua_State *L, Table *t, unsigned int nasize, unsigned int nhsize);

static void rehash (lua_State *L, Table *t, const TValue *ek);
static Node *getfreepos (Table *t);
const TValue *luaH_get (Table *t, const TValue *key);
static Node *mainpositionfromnode (const Table *t, Node *nd);

static void luaH_newkey (lua_State *L, Table *t, const TValue *key, TValue *value) {
  Node *mp;
  TValue aux;
  if ((key->tt_ & 0xf) == 0)
    luaG_runerror(L, "table index is nil");
  else if (key->tt_ == 0x13) {
    lua_Number f = key->value_.n;
    lua_Integer k;
    if (luaV_flttointeger(f, &k, 0)) {
      aux.value_.i = k;
      aux.tt_ = 0x03;
      key = &aux;
    }
    else if (f != f)
      luaG_runerror(L, "table index is NaN");
  }
  if ((value->tt_ & 0xf) == 0)
    return;
  mp = mainpositionTV(t, key);
  if ((mp->i_val.tt_ & 0xf) != 0 || t->lastfree == NULL) {
    Node *othern;
    Node *f = getfreepos(t);
    if (f == NULL) {
      rehash(L, t, key);
      {
        const TValue *slot = luaH_get(t, key);
        if (slot->tt_ == 0x20)
          luaH_newkey(L, t, key, value);
        else {
          TValue *s = (TValue *)slot;
          s->value_ = value->value_;
          s->tt_ = value->tt_;
        }
      }
      return;
    }
    othern = mainpositionfromnode(t, mp);
    if (othern != mp) {
      while (othern + othern->u.next != mp)
        othern += othern->u.next;
      othern->u.next = (int)(f - othern);
      *f = *mp;
      if (mp->u.next != 0) {
        f->u.next += (int)(mp - f);
        mp->u.next = 0;
      }
      mp->i_val.tt_ = 0x10;
    }
    else {
      if (mp->u.next != 0)
        f->u.next = (int)((mp + mp->u.next) - f);
      mp->u.next = (int)(f - mp);
      mp = f;
    }
  }
  mp->u.key_val = key->value_;
  mp->u.key_tt = key->tt_;
  if ((key->tt_ & 0x40) && (t->marked & (1 << 5)) && (key->value_.gc->marked & (3 << 3)))
    luaC_barrierback_(L, (GCObject *)t);
  mp->i_val.value_ = value->value_;
  mp->i_val.tt_ = value->tt_;
}
