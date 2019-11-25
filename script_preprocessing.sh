#!/bin/bash

date=$1
echo $date
vd="${date}_VoterDetail"
echo $vd
if [ ! -d preprocessed/$date ]; then
  mkdir -p preprocessed/$date;
fi
for file in data/$vd/*; do
	f="$(basename "$file")"
	county="$(echo $f  | cut -d'_' -f1)"
	python florida_preprocessing_pipeline.py $county $date
done
