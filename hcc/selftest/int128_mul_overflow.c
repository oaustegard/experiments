// EXPECT-EXIT: 0
// __builtin_mul_overflow on __int128 operands: signed/unsigned domains, boundary
// products, and narrower result types (the fit check is against the result type).
typedef __int128 i128;
typedef unsigned __int128 u128;
int main (void)
{
  i128 big = (i128) 1 << 64;
  i128 r;
  int bad = 0;
  if (__builtin_mul_overflow (big, big, &r) != 1) bad |= 1;             /* 2^128 overflows i128 */
  if (__builtin_mul_overflow ((i128) 1 << 62, 2, &r) != 0) bad |= 2;    /* 2^63 fits in i128 */
  if (r != (i128) 1 << 63) bad |= 4;
  if (__builtin_mul_overflow (-big, (i128) 1 << 63, &r) != 0 || r != -((i128) 1 << 127)) bad |= 8;
  if (__builtin_mul_overflow (-big, big, &r) != 1) bad |= 512;
  long long small;
  if (__builtin_mul_overflow (big, 3, &small) != 1) bad |= 16;         /* 3*2^64 does not fit long long */
  if (__builtin_mul_overflow ((i128) -3, 5, &small) != 0 || small != -15) bad |= 32;
  u128 ur;
  if (__builtin_mul_overflow ((u128) big, (u128) big, &ur) != 1) bad |= 64;
  if (__builtin_mul_overflow ((u128) big - 1, (u128) big + 1, &ur) != 0 || ur != ~(u128) 0) bad |= 128;
  if (__builtin_mul_overflow ((u128) big - 1, (u128) 1, &ur) != 0 || ur != (u128) big - 1) bad |= 256;
  return bad;
}
