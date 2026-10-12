// EXPECT-EXIT: 3
// Compatible file-scope redeclarations (VLA-free array params, incomplete array) are accepted.
extern int arr[];
int arr[2] = { 1, 2 };
int f(int a[], int);
int f(int *p, int n) { return p[0] + p[1] + n - 1; }
int main(void) { return f(arr, 1) ; }
