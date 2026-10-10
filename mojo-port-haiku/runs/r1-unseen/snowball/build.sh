#!/bin/sh
set -e
cd "$(dirname "$0")"
mojo build msnowball/kernel.mojo --emit shared-lib -o msnowball/_kernel.so
