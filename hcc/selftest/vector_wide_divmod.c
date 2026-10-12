// EXPECT-EXIT: 0
typedef unsigned __int128 U;
typedef __int128 S;
typedef unsigned __int128 UV __attribute__((vector_size(32)));
typedef __int128 SV __attribute__((vector_size(32)));
UV f1(UV a, UV b){ return a / b; }
UV f2(UV a, UV b){ return a % b; }
SV f3(SV a, SV b){ return a / b; }
SV f4(SV a, SV b){ return a % b; }
int main(){
  UV a = {100, (U)1<<100}, b = {7, 3};
  UV q = f1(a,b), r = f2(a,b);
  SV c = {-100, -((S)1<<100)}, d = {7, -3};
  SV q2 = f3(c,d), r2 = f4(c,d);
  return !(q[0]==14 && r[0]==2 && q[1]==((U)1<<100)/3 && r[1]==((U)1<<100)%3 && q2[0]==-14 && r2[0]==-2 && q2[1]==((S)-((S)1<<100))/-3 && r2[1]==-(((S)1<<100)%3));
}
