#!/bin/bash
date=$1
vd="${date}_VoterDetail"
if [ ! -d couples/$date ]; then
	mkdir -p couples/$date;
	echo "created new dir"
fi
counter=0
for file in data/$vd/*; do
    f="$(basename "$file")"
    county="$(echo $f | cut -d'_' -f1)"
    counter=$(($counter+1))
    # echo $county
    # echo $counter
    # echo $(($counter%9))
    if [ $(($counter%10)) -eq 0 ]
        then
                wait
    fi
    # echo "couples/$date/couples_"$county"_$date.csv";
    if [ ! -f "couples/$date/couples_"$county"_$date.csv" ]; then
		echo "Will process county $county";
		python florida_couples_pipeline.py ${county} ${date} &
    fi
done
