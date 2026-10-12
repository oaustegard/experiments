// EXPECT-EXIT: 0
int f(void) {
  static const unsigned short arr[] = (const unsigned short []){0x0D2B, 7};
  return arr[0] == 0x0D2B && arr[1] == 7 ? 0 : 1;
}
int main(void) { return f(); }
