if [ "$#" -ne 1 ]; then
    echo "Usage ${0} logfile"
    exit
fi

grep $(grep finish $1 | cut -d" " -f13 | sort -g | tail -1) $1
