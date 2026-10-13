static const char udatatypename[] = "userdata";

const char *const luaT_typenames_[] = {  /* ORDER LUA_T */
  "no value",
  "nil", "boolean", udatatypename, "number",
  "string", "table", "function", udatatypename, "thread",
  "upvalue", "proto" /* these last cases are used for tests only */
};
