// EXPECT-STDOUT: hello, world 42 -7 ff 3.50
// EXPECT-STDOUT: str=abc c=x
#include <stdio.h>
int main(void) {
    printf("hello, %s %d %ld %x %.2f\n", "world", 42, -7L, 255, 3.5);
    printf("str=%s c=%c\n", "abc", 'x');
    return 0;
}
