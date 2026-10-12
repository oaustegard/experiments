// EXPECT-EXIT: 10
// C23 enum with a fixed underlying type: `enum e : type { ... }` and a
// forward-declared `enum g : long;` completed later. Size follows the type.
enum e : _Bool { X };
enum f : unsigned char { Y = 200 };
enum g : long;
enum g : long { Z };
struct S { enum f : 3; int a : 2; };
int main(void) {
    _Bool b = 1;
    return sizeof(enum e) + sizeof(enum f) + sizeof(enum g) + X + Z + b - 1;
}
