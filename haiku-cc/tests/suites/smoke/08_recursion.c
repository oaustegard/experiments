int fib(int n) { return n < 2 ? n : fib(n-1) + fib(n-2); }
int main(void) { return fib(10); }
