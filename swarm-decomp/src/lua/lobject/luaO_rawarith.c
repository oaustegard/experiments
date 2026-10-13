#include "fn_luaO_rawarith.h"
#include "fn_luaV_tointegerns.h"

#define ttisfloat(o) ((o)->tt_ == 0x13)
#define setfltvalue(obj,x) { TValue *io_ = (obj); io_->value_.n = (x); io_->tt_ = 0x13; }
#define fltvalue(o) ((o)->value_.n)
#define l_likely(x) (__builtin_expect(((x) != 0), 1))
#define tointegerns(o,i) (l_likely(ttisinteger(o)) ? (*(i) = ivalue(o), 1) : luaV_tointegerns(o,i,0))
#define tonumberns(o,n) (ttisfloat(o) ? ((n) = fltvalue(o), 1) : (ttisinteger(o) ? ((n) = (lua_Number)ivalue(o), 1) : 0))

int luaO_rawarith (lua_State *L, int op, const TValue *p1, const TValue *p2, StkId res) {
  switch (op) {
    case 7: case 8: case 9: case 10: case 11: case 13: {
      lua_Integer i1; lua_Integer i2;
      if (tointegerns(p1, &i1) && tointegerns(p2, &i2)) {
        setivalue(res, intarith(L, op, i1, i2));
        return 1;
      }
      else return 0;
    }
    case 4: case 5: {
      lua_Number n1; lua_Number n2;
      if (tonumberns(p1, n1) && tonumberns(p2, n2)) {
        setfltvalue(res, numarith(L, op, n1, n2));
        return 1;
      }
      else return 0;
    }
    default: {
      lua_Number n1; lua_Number n2;
      if (ttisinteger(p1) && ttisinteger(p2)) {
        setivalue(res, intarith(L, op, ivalue(p1), ivalue(p2)));
        return 1;
      }
      else if (tonumberns(p1, n1) && tonumberns(p2, n2)) {
        setfltvalue(res, numarith(L, op, n1, n2));
        return 1;
      }
      else return 0;
    }
  }
}
