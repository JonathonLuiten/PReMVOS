if [ "$#" -ne 1 ]; then
    echo "Usage ${0} logfile"
    exit
fi

config=$1
config2=$(echo ${config} | sed s/trainsplit/trainsplit2/g)
config3=$(echo ${config} | sed s/trainsplit/trainsplit3/g)


if [ ! -e $config ]; then
  echo "${config} does not exist!"
  exit 1
fi

cp ${config} ${config2}
sed -i s/_trainsplit/_trainsplit2/g ${config2}
sed -i s/"trainsplit\": 1,"/"trainsplit\": 2,"/g ${config2}

cp ${config} ${config3}
sed -i s/_trainsplit/_trainsplit3/g ${config3}
sed -i s/"trainsplit\": 1,"/"trainsplit\": 3,"/g ${config3}
