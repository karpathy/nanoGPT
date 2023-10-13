#!/bin/bash

declare -a bases=("1" "2" "4" "8" "16")

data_dir="data"

if [ ! -d "${data_dir}" ]; then
  mkdir -p "${data_dir}"
fi

for i in "${bases[@]}"; do
  echo "$i"
  python print_mod_16.py --base "$i" --seed 16 > "./${data_dir}/base_${i}.txt"
done


