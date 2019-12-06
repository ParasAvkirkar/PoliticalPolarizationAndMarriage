#!/bin/bash
date=$1
county_file_directory="data/NewYork/${date}_county_files"
echo "Processing counties in directory $county_file_directory"
counter=0
for file in $county_file_directory/*; do
#     f="$(basename "$file")"
    county="$(echo $file | cut -d'_' -f 5 | cut -d'.' -f 1)"
#     echo $county
    counter=$(($counter+1))
    # echo $county
    # echo $counter
    # echo $(($counter%9))
    if [ $(($counter%10)) -eq 0 ]
        then
                wait
    fi
    # echo "couples/$date/couples_"$county"_$date.csv";
#     echo "$county_file_directory/county_${date}_${county}.csv"
#     echo "data/NewYork/couples/$date/couples_${date}_${county}.csv"
    if [ ! -f "data/NewYork/couples/$date/couples_${date}_${county}.csv" ]; then
		echo "Will process county $county";
		python new_york_couples_pipeline.py ${county} ${date} &
    fi
done

