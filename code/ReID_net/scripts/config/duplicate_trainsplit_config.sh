if [ "$#" -ne 1 ]; then
    echo "Usage ${0} logfile (without trainsplit)"
    exit
fi

NEW_IDX=2

for split in "" "2" "3"; do
  cp ${1}_trainsplit${split} ${1}_${NEW_IDX}_trainsplit${split}
  sed -i s/${1}_trainsplit${split}/${1}_${NEW_IDX}_trainsplit${split}/g ${1}_${NEW_IDX}_trainsplit${split}
done
