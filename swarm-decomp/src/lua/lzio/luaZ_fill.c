
#define EOZ (-1)

int luaZ_fill (ZIO *z) {
  size_t size;
  lua_State *L = z->L;
  const char *buff;
  buff = z->reader(L, z->data, &size);
  if (buff == NULL || size == 0)
    return EOZ;
  z->n = size - 1;
  z->p = buff;
  return (unsigned char)*(z->p++);
}
