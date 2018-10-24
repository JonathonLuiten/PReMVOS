if [ "$#" -lt 1 ]; then
    echo "Usage ${0} logfile(s)"
    exit
fi

#grep iou $1 | grep sequence | cut -d"'" -f5| cut -d" " -f2 | cut -d"," -f1 | ../scripts/stats/print_stats.py
grep -h iou $@ | grep sequence | cut -d"'" -f5| cut -d" " -f2 | cut -d"," -f1 | ../scripts/stats/print_stats.py
