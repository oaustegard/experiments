// EXPECT-OK
void f(int);
void f(x) unsigned char x; {}
int main(void){ f(1); return 0; }
