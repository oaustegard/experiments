// EXPECT-FLAGS: -fshort-wchar
// EXPECT-EXIT: 0
// -fshort-wchar: wchar_t is unsigned short, L'x' and L"..." use 16-bit units.
extern void abort(void);
typedef __WCHAR_TYPE__ W;
int main(void)
{
  W w = ~(W) 0;
  if (w != __WCHAR_MAX__ || sizeof(W) != 2 || sizeof(L'a') != 2) abort();
  if (sizeof(L"ab") != 6 || L"xy"[1] != 'y') abort();
  return 0;
}
