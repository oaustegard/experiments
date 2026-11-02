#!/bin/sh
# Kodak test images used by the sweep (768x512 PNG, from Rich Franzen's mirror of the Kodak set).
cd "$(dirname "$0")" && mkdir -p kodak && for i in 01 05 07 15 20 23; do
  [ -f kodak/kodim$i.png ] || curl -sSfL -o kodak/kodim$i.png https://r0k.us/graphics/kodak/kodak/kodim$i.png
done
