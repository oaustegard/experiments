// EXPECT-EXIT: 0
// An enum type is compatible with its underlying integer type (int here, since
// it has a negative enumerator), so a selector of type int matches `enum E:`.
enum E { E1 = -1, E2 = 0 };
typedef int A;
typedef enum E B;
_Static_assert(_Generic((A){ 0 }, B: 1), "");
_Static_assert(_Generic((B)0, A: 1, default: 0), "");
int main(void) { return 0; }
