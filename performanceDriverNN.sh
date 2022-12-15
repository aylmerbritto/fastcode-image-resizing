#! /bin/sh

fileList=$(ls -v inputs/black-images)
for i in $fileList;do
    size=$(echo $i | grep -Eo '[0-9]{1,4}' | head -1)
    ./build/performanceNN inputs/black-images/$i $size $size 
    ./build/bmnn inputs/black-images/$i
done