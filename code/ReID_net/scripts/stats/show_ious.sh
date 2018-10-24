if [ "$#" -ne 1 ]; then
    echo "Usage ${0} logfile"
    exit
fi

grep iou $1 | grep sequence | cut -d"'" -f5| cut -d" " -f2 | cut -d"," -f1
