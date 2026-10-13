static int str_format (lua_State *L) {
  int top = lua_gettop(L);
  int arg = 1;
  size_t sfl;
  const char *strfrmt = luaL_checklstring(L, arg, &sfl);
  const char *strfrmt_end = strfrmt+sfl;
  const char *flags;
  luaL_Buffer b;
  luaL_buffinit(L, &b);
  while (strfrmt < strfrmt_end) {
    if (*strfrmt != L_ESC)
      luaL_addchar(&b, *strfrmt++);
    else if (*++strfrmt == L_ESC)
      luaL_addchar(&b, *strfrmt++);
    else {
      char form[MAX_FORMAT];
      int maxitem = MAX_ITEM;
      char *buff = luaL_prepbuffsize(&b, maxitem);
      int nb = 0;
      if (++arg > top)
        return luaL_argerror(L, arg, "no value");
      {
        size_t len = strspn(strfrmt, L_FMTFLAGSF "123456789.");
        char *f = form;
        len++;
        if (len >= MAX_FORMAT - 10)
          luaL_error(L, "invalid format (too long)");
        *(f++) = '%';
        memcpy(f, strfrmt, len * sizeof(char));
        *(f + len) = '\0';
        strfrmt = strfrmt + len - 1;
      }
      switch (*strfrmt++) {
        case 'c': {
          checkformat(L, form, L_FMTFLAGSC, 0);
          nb = snprintf(buff, maxitem, form, (int)luaL_checkinteger(L, arg));
          break;
        }
        case 'd': case 'i':
          flags = L_FMTFLAGSI;
          goto intcase;
        case 'u':
          flags = L_FMTFLAGSU;
          goto intcase;
        case 'o': case 'x': case 'X':
          flags = L_FMTFLAGSX;
         intcase: {
          lua_Integer n = luaL_checkinteger(L, arg);
          checkformat(L, form, flags, 1);
          addlenmod(form, "ll");
          nb = snprintf(buff, maxitem, form, (LUAI_UACINT)n);
          break;
        }
        case 'a': case 'A':
          checkformat(L, form, L_FMTFLAGSF, 1);
          addlenmod(form, "");
          nb = snprintf(buff, maxitem, form, luaL_checknumber(L, arg));
          break;
        case 'f': case 'F':
          maxitem = MAX_ITEMF;
          buff = luaL_prepbuffsize(&b, maxitem);
          /* FALLTHROUGH */
        case 'e': case 'E': case 'g': case 'G': {
          lua_Number n = luaL_checknumber(L, arg);
          checkformat(L, form, L_FMTFLAGSF, 1);
          addlenmod(form, "");
          nb = snprintf(buff, maxitem, form, (LUAI_UACNUMBER)n);
          break;
        }
        case 'p': {
          const void *p = lua_topointer(L, arg);
          checkformat(L, form, L_FMTFLAGSC, 0);
          if (p == NULL) {
            p = "(null)";
            form[strlen(form) - 1] = 's';
          }
          nb = snprintf(buff, maxitem, form, p);
          break;
        }
        case 'q': {
          if (form[2] != '\0')
            return luaL_error(L, "specifier '%%q' cannot have modifiers");
          switch (lua_type(L, arg)) {
            case LUA_TSTRING: {
              size_t len;
              const char *s = lua_tolstring(L, arg, &len);
              addquoted(&b, s, len);
              break;
            }
            case LUA_TNUMBER: {
              char *bq = luaL_prepbuffsize(&b, MAX_ITEM);
              if (!lua_isinteger(L, arg))
                nb = quotefloat(L, bq, lua_tonumber(L, arg));
              else {
                lua_Integer nq = lua_tointeger(L, arg);
                const char *fmtq = (nq == LUA_MININTEGER)
                                 ? "0x%llx" : "%lld";
                nb = snprintf(bq, MAX_ITEM, fmtq, (LUAI_UACINT)nq);
              }
              luaL_addsize(&b, nb);
              break;
            }
            case LUA_TNIL: case LUA_TBOOLEAN: {
              luaL_tolstring(L, arg, NULL);
              luaL_addvalue(&b);
              break;
            }
            default: {
              luaL_argerror(L, arg, "value has no literal form");
            }
          }
          break;
        }
        case 's': {
          size_t l;
          const char *s = luaL_tolstring(L, arg, &l);
          if (form[2] == '\0')
            luaL_addvalue(&b);
          else {
            luaL_argcheck(L, l == strlen(s), arg, "string contains zeros");
            checkformat(L, form, L_FMTFLAGSC, 1);
            if (strchr(form, '.') == NULL && l >= 100) {
              luaL_addvalue(&b);
            }
            else {
              nb = snprintf(buff, maxitem, form, s);
              lua_pop(L, 1);
            }
          }
          break;
        }
        default: {
          return luaL_error(L, "invalid conversion '%s' to 'format'", form);
        }
      }
      luaL_addsize(&b, nb);
    }
  }
  luaL_pushresult(&b);
  return 1;
}
