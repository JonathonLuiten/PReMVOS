#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage ${0} model"
    exit
fi
MODEL=$1

(
  cd "/home/${USER}/vision/savitar/forwarded/${MODEL}"
  det_jsons=$(ls det_*.json)
  for det_json in ${det_jsons}; do
    cd "/home/${USER}/vision/savitar/"
    det=$(echo ${det_json} | sed "s/.json//g")
    echo "converting ${det}..."
    ./scripts/postproc/convert_coco_detections_to_kitti_format.py ${MODEL} ${det} | grep -v "warning, ignoring detection with unknown class"
    cd "/home/$USER/vision/KITTI_object_devkit"
    echo "evaluating ${det}..."
    ./cpp/evaluate_object ${MODEL}
    cd "/home/${USER}/vision/savitar/forwarded/${MODEL}"
    echo "=========="
    mv stats_car_detection.txt ${det}_car.txt
    mv stats_pedestrian_detection.txt ${det}_pedestrian.txt
    mv stats_cyclist_detection.txt ${det}_cyclist.txt
  done
)


#now collect all the results
cd "/home/${USER}/vision/savitar/forwarded/${MODEL}"
(
for class in car pedestrian cyclist; do
  epochs=$(ls det*_${class}.txt | sed s/det_//g | sed s/_${class}.txt//g | sort -n)
  for epoch in $epochs; do
    file=$(echo det_${epoch}_${class}.txt)
    echo $file | sed "s/det_//g" | sed "s/_${class}//g" | sed "s/.txt//g"
  done | tr '\n' ' '
  #result for easy
  printf "\n${class} easy\n"
  for epoch in $epochs; do
    file=$(echo det_${epoch}_${class}.txt)
    sed '2q;d' $file | sed "s/ap //g"
  done | tr '\n' ' '
  #result for medium
  printf "\n${class} medium\n"
  for epoch in $epochs; do
    file=$(echo det_${epoch}_${class}.txt)
    sed '4q;d' $file | sed "s/ap //g"
  done | tr '\n' ' '
  #result for hard
  printf "\n${class} hard\n"
  for epoch in $epochs; do
    file=$(echo det_${epoch}_${class}.txt)
    sed '6q;d' $file | sed "s/ap //g"
  done | tr '\n' ' '
  echo
done
) | tee results.txt

cp results.txt /home/voigtlaender/pub/cvpr2018/results/detector/eval/${MODEL}
