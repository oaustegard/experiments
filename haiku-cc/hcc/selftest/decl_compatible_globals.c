// EXPECT-EXIT: 0
// Compatible redeclarations at file scope and in block scope are valid.
extern int x;
int x = 1;
int arr[];
int arr[3];
typedef int t;
typedef int t;
struct s;
extern struct s *sp;
struct s { int m; } *sp;
int main(void) {
    extern int x;
    extern int arr[3];
    typedef int t;
    return x + arr[0] + sizeof(t) - 5;
}
