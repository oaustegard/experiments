// EXPECT-EXIT: 0
int f(int n; int m, int n) { return m * 10 + n; }
int g(int n; int a[n], int n) { return a[n-1]; }
int main(void) {
  int arr[3] = {7, 8, 9};
  if (f(4, 2) != 42) return 1;
  if (g(arr, 3) != 9) return 2;
  return 0;
}
