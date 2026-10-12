/* { dg-do compile } */
/* { dg-options "-O2 -fdump-tree-optimized -fno-trapping-math" } */
/* PR tree-optimization/107881 */

_Bool ltgt_eq(float a, float b)
{
  _Bool c = a < b;
  _Bool d = a > b;
  return c == d; // a u== b
}
/* { dg-final { scan-tree-dump "a_\[0-9\]+.D. u== b_\[0-9\]+.D.|b_\[0-9\]+.D. u== a_\[0-9\]+.D." "optimized" } } */

_Bool lteq_eq(float x, float y)
{
  _Bool c = x < y;
  _Bool d = x == y;
  return c == d; // x u> y
}
/* { dg-final { scan-tree-dump "x_\[0-9\]+.D. u> y_\[0-9\]+.D.|y_\[0-9\]+.D. u< x_\[0-9\]+.D." "optimized" } } */

_Bool ltne_eq(float z, float w)
{
  _Bool c = z < w;
  _Bool d = z != w;
  return c == d; // z <= w
}
/* { dg-final { scan-tree-dump "z_\[0-9\]+.D. <= w_\[0-9\]+.D.|w_\[0-9\]+.D. >= y_\[0-9\]+.D." "optimized" } } */

_Bool lege_eq(float i, float j)
{
  _Bool c = i <= j;
  _Bool d = i >= j;
  return c == d; // i u== j
}
/* { dg-final { scan-tree-dump "i_\[0-9\]+.D. u== j_\[0-9\]+.D.|j_\[0-9\]+.D. u== i_\[0-9\]+.D." "optimized" } } */

_Bool leeq_eq(float k, float l)
{
  _Bool c = k <= l;
  _Bool d = k == l;
  return c == d; // k u>= l
}
/* { dg-final { scan-tree-dump "k_\[0-9\]+.D. u>= l_\[0-9\]+.D.|l_\[0-9\]+.D. u<= k_\[0-9\]+.D." "optimized" } } */

_Bool lene_eq(float m, float n)
{
  _Bool c = m <= n;
  _Bool d = m != n;
  return c == d; // m < n
}
/* { dg-final { scan-tree-dump "m_\[0-9\]+.D. < n_\[0-9\]+.D.|n_\[0-9\]+.D. > m_\[0-9\]+.D." "optimized" } } */
