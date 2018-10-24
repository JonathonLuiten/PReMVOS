if [ "$#" -ne 1 ]; then
    echo "Usage ${0} model"
    exit
fi

for seq in blackswan bmx-trees breakdance camel car-roundabout car-shadow cows dance-twirl dog drift-chicane drift-straight goat horsejump-high kite-surf libby motocross-jump paragliding-launch parkour scooter-black soapbox; do
  echo $seq
  RES1=$(grep $seq $1.log | grep iou | cut -d"'" -f5| cut -d" " -f2 | cut -d"," -f1)
  RES2=$(grep $seq $1_2.log | grep iou | cut -d"'" -f5| cut -d" " -f2 | cut -d"," -f1)
  RES3=$(grep $seq $1_3.log | grep iou | cut -d"'" -f5| cut -d" " -f2 | cut -d"," -f1)
  echo "$RES1 $RES2 $RES3" | ../scripts/print_mean_std_ddof_pretty.py
done
