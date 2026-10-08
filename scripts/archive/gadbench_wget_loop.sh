#!/bin/bash
ID="1txzXrzwBBAOEATXmfKzMUUKaXh6PJeR1"
URL="https://drive.usercontent.google.com/download?id=${ID}&export=download"
for i in $(seq 1 40); do
  sz=$(stat -c%s gadbench_datasets.zip 2>/dev/null || echo 0)
  if [ "$sz" -ge 790000000 ]; then echo "COMPLETE size=$sz"; break; fi
  wget -c -q --load-cookies /tmp/cookies.txt "${URL}" -O gadbench_datasets.zip >> wget.log 2>&1
  echo "loop $i size=$(stat -c%s gadbench_datasets.zip 2>/dev/null)"
done
