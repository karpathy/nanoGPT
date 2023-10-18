#!/bin/bash

declare -a bases=("1" "2" "4" "8" "16")

data_dir="data"

if [ ! -d "${data_dir}" ]; then
  mkdir -p "${data_dir}"
fi

for i in "${bases[@]}"; do
  echo "$i"
  python print_bases_mod_x.py --modulo 128 --no_separator --base "$i" --seed 16 > "./${data_dir}/base_${i}.txt"
  # python print_bases_mod_x.py --modulo 128  --base "$i" --seed 16 > "./${data_dir}/base_${i}.txt"
done


