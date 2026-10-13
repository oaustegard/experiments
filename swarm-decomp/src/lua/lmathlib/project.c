static lua_Unsigned project (lua_Unsigned ran, lua_Unsigned n, RanState *state) {
  if ((n & (n + 1)) == 0)  /* is 'n + 1' a power of 2? */
    return ran & n;  /* no bias */
  else {
    lua_Unsigned lim = n;
    /* compute the smallest (2^b - 1) not smaller than n */
    lim |= (lim >> 1);
    lim |= (lim >> 2);
    lim |= (lim >> 4);
    lim |= (lim >> 8);
    lim |= (lim >> 16);
#if (LUA_MAXUNSIGNED >> 31) >= 3
    lim |= (lim >> 32);  /* integer type has more than 32 bits */
#endif
    while ((ran &= lim) > n)  /* project 'ran' into [0, lim] */
      ran = nextrand(state->s);  /* not inside [0, n]? Try again */
    return ran;
  }
}
