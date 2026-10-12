// EXPECT-EXIT: 0
// A designator range applies its value to every index, and the brace-elided
// values after it continue once, after the last index (not once per index).
// A member designator's brace-elided tail continues with the next member.
struct I { int J; int K[3]; int L; };
struct M { int N; struct I O[3]; int P; };
struct M n[] = { [0 ... 5].O[1 ... 2].K[0 ... 1] = 4, 5, 6, 7 };
struct S { int a; int b; int c; } s = { .b = 1, 2 };
int main(void)
{
  if (n[0].O[1].K[1] != 4 || n[5].O[2].K[1] != 4) return 1;
  if (n[0].O[1].K[2] != 0 || n[4].O[2].K[2] != 0) return 2;
  if (n[5].O[2].K[2] != 5 || n[5].O[2].L != 6 || n[5].P != 7) return 3;
  if (s.a != 0 || s.b != 1 || s.c != 2) return 4;
  return 0;
}
