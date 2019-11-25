#!/bin/bash
date=$1
echo $date
vd="${date}_VoterDetail"
echo $vd
if [ ! -d couples/$date ]; then
	mkdir -p couples/$date;
fi
counter=0
for file in data/$vd/*; do
	f="$(basename "$file")"
	county="$(echo $f  | cut -d'_' -f1)"
    counter=$(($counter+1))
    # echo $counter
    # echo $(($counter%7))
    if [ $(($counter%7)) -eq 0 ]
    	then
        	wait
    fi
    python florida_couples_pipeline.py ${county} ${date} &
    
done
