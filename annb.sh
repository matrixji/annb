#!/usr/bin/env bash

this_dir=$(cd $(dirname $0); pwd)
PYTHONPATH=$this_dir:$PYTHONPATH python3 -m annb.cli "$@"
