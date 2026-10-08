#!/bin/bash
source /home/lixin/miniconda3/bin/activate CBP
ID="1txzXrzwBBAOEATXmfKzMUUKaXh6PJeR1"
for i in $(seq 1 40); do
  sz=$(stat -c%s gadbench_datasets.zip 2>/dev/null || echo 0)
  if [ "$sz" -ge 790000000 ]; then echo "COMPLETE size=$sz"; break; fi
  gdown --continue "$ID" -O gadbench_datasets.zip >> gdown.log 2>&1
  echo "loop $i size=$(stat -c%s gadbench_datasets.zip 2>/dev/null)"
done
