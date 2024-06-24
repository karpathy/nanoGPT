#!/bin/bash

# Create datasets for different movements:
# will pvl be learned more quickly after learnign udf?
# will all be learned more quickly after udf and pvl?
# will dataset mixtures of prior action sets be necessary to prevent forgetting?
python3 create_dataset.py -s 10000 -m 10000 -a u d f -t 1000 -c --output test_udf.txt --charlist
python3 create_dataset.py -s 10000 -m 10000 -a p v l -t 1000 -c --output test_pvl.txt --charlist
python3 create_dataset.py -s 10000 -m 10000 -t 1000 -c --output test_all.txt --charlist

