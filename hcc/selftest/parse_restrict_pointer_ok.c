// restrict on an object pointer is valid
typedef char *P;
restrict P ok;
char *restrict p = 0;
int main(void) { return p != 0; }
