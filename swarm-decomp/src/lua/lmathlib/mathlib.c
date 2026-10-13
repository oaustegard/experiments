static int math_abs (lua_State *L);
static int math_acos (lua_State *L);
static int math_asin (lua_State *L);
static int math_atan (lua_State *L);
static int math_ceil (lua_State *L);
static int math_cos (lua_State *L);
static int math_deg (lua_State *L);
static int math_exp (lua_State *L);
static int math_toint (lua_State *L);
static int math_floor (lua_State *L);
static int math_fmod (lua_State *L);
static int math_ult (lua_State *L);
static int math_log (lua_State *L);
static int math_max (lua_State *L);
static int math_min (lua_State *L);
static int math_modf (lua_State *L);
static int math_rad (lua_State *L);
static int math_sin (lua_State *L);
static int math_sqrt (lua_State *L);
static int math_tan (lua_State *L);
static int math_type (lua_State *L);
static int math_cosh (lua_State *L);
static int math_sinh (lua_State *L);
static int math_tanh (lua_State *L);
static int math_pow (lua_State *L);
static int math_frexp (lua_State *L);
static int math_ldexp (lua_State *L);
static int math_log10 (lua_State *L);

static const luaL_Reg mathlib[] = {
  {"abs", math_abs},
  {"acos", math_acos},
  {"asin", math_asin},
  {"atan", math_atan},
  {"ceil", math_ceil},
  {"cos", math_cos},
  {"deg", math_deg},
  {"exp", math_exp},
  {"tointeger", math_toint},
  {"floor", math_floor},
  {"fmod", math_fmod},
  {"ult", math_ult},
  {"log", math_log},
  {"max", math_max},
  {"min", math_min},
  {"modf", math_modf},
  {"rad", math_rad},
  {"sin", math_sin},
  {"sqrt", math_sqrt},
  {"tan", math_tan},
  {"type", math_type},
  {"atan2", math_atan},
  {"cosh", math_cosh},
  {"sinh", math_sinh},
  {"tanh", math_tanh},
  {"pow", math_pow},
  {"frexp", math_frexp},
  {"ldexp", math_ldexp},
  {"log10", math_log10},
  {"random", NULL},
  {"randomseed", NULL},
  {"pi", NULL},
  {"huge", NULL},
  {"maxinteger", NULL},
  {"mininteger", NULL},
  {NULL, NULL}
};
