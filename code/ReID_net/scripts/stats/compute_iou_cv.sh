if [ "$#" -ne 1 ]; then
    echo "Usage ${0} logfile"
    exit
fi

F1=$1
F2=$(echo $1 | sed s/trainsplit/trainsplit2/g)
F3=$(echo $1 | sed s/trainsplit/trainsplit3/g)

grep -h iou $F1 $F2 $F3 | grep sequence | cut -d"'" -f5| cut -d" " -f2 | cut -d"," -f1 | ../scripts/print_stats.py
