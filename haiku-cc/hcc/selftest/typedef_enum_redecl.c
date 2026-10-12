// EXPECT-ERROR: conflicting types for 'TE'
// Typedef redeclaration must name the same type: enum e is not the same type as int.
enum e { E1 = 0 };
typedef enum e TE;
typedef int TE;
int main(void) { return 0; }
