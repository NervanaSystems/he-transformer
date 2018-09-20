#!/bin/bash

NGRAPH_ENABLE_SERIALIZE=1 python test_squash.py --batch_size=1
NGRAPH_ENABLE_SERIALIZE=1 python test_squash.py --batch_size=2
NGRAPH_ENABLE_SERIALIZE=1 python test_squash.py --batch_size=4
NGRAPH_ENABLE_SERIALIZE=1 python test_squash.py --batch_size=8
NGRAPH_ENABLE_SERIALIZE=1 python test_squash.py --batch_size=16
NGRAPH_ENABLE_SERIALIZE=1 python test_squash.py --batch_size=32
NGRAPH_ENABLE_SERIALIZE=1 python test_squash.py --batch_size=64
NGRAPH_ENABLE_SERIALIZE=1 python test_squash.py --batch_size=128
NGRAPH_ENABLE_SERIALIZE=1 python test_squash.py --batch_size=256
NGRAPH_ENABLE_SERIALIZE=1 python test_squash.py --batch_size=512
NGRAPH_ENABLE_SERIALIZE=1 python test_squash.py --batch_size=1024
NGRAPH_ENABLE_SERIALIZE=1 python test_squash.py --batch_size=2048
NGRAPH_ENABLE_SERIALIZE=1 python test_squash.py --batch_size=4096
