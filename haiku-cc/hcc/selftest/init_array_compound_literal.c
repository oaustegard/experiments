// EXPECT-EXIT: 0
int a[] = (int[4]){1, 1, 2};
int b[3] = (int[3]){4, 5, 6};
int main(void) {
  int sum = 0;
  for (int i = 0; i < 4; i++) sum += a[i];
  if (sum != 4 || (sizeof a) != 4 * sizeof(int)) return 1;
  if (b[1] != 5) return 2;
  return 0;
}
