// EXPECT-EXIT: 10
#define X 1
#pragma push_macro("X")
#undef X
#define X 2
#pragma pop_macro("X")
#pragma push_macro("Y")
#define Y 5
#pragma pop_macro("Y")
int main(void)
{
  return X * 10
#ifdef Y
    + 1
#endif
  ;
}
