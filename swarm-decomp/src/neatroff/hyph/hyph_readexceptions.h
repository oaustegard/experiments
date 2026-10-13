#pragma once

/* the address of a string literal passed as a plain pointer lets gcc -O2
   (IPA-CP) make a constprop clone and drop the readers; an integer parameter
   keeps the callers' calls to the readers themselves */
static void hyph_readpatterns(long s0);
static void hyph_readexceptions(long s0);
