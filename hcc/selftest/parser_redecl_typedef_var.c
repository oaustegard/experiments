// EXPECT-EXIT: 0
// Enum tag and typedef of the same name are different namespaces; a typedef
// and a variable of one name in the same scope are not.
typedef enum a { XYZ } a;
struct a { int x; };
int main(void) { return 0; }
