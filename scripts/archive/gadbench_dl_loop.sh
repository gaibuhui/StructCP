#!/bin/bash
source /home/lixin/miniconda3/bin/activate CBP
for i in $(seq 1 30); do
  if [ -f gadbench_datasets.zip ]; then
    sz=$(stat -c%s gadbench_datasets.zip)
    if [ "$sz" -ge 790000000 ]; then echo "COMPLETE size=$sz"; break; fi
  fi
  gdown --continue "1txzXrzwBBAOEATXmfKzMUUKaXh6PJeR1" -O gadbench_datasets.zip >> download.log 2>&1
  echo "loop $i done, size=$(stat -c%s gadbench_datasets.zip 2>/dev/null)"
done
