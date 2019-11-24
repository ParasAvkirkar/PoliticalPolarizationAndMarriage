#!/bin/bash

date=$1
echo $date
vd="${date}_VoterDetail"
echo $vd

if [ ! -d couples/$date ]; then
  mkdir -p couples/$date;
fi

for file in data/$vd/*; do
	f="$(basename "$file")"
	county="$(echo $f  | cut -d'_' -f1)"
	python florida_couples_pipeline.py ${county} ${date} &
	
done
