// EXPECT-EXIT: 0
// K&R parameter list may contain empty declarations (`int;`), which are ignored.
void f(a, b)
     int;
     register;
     int a; char b;
{
}
int main(void) {
    f(1, 'x');
    return 0;
}
