// EXPECT-EXIT: 0
// #pragma pack inside a struct body applies to the members declared after it only.
struct S { char a[3];
#pragma pack(1)
  struct T { char b; int c; } d;
#pragma pack()
  int e; };
int s1[(sizeof(struct S) == 12) ? 1 : -1];
int s2[(sizeof(struct T) == 5) ? 1 : -1];
int main(void) { return 0; }
