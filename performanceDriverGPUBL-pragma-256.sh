#! /bin/sh

fileList=$(ls -v inputs/black-images)
for i in $fileList;do
    size=$(echo $i | grep -Eo '[0-9]{1,4}' | head -1)
    ./build/gpu-bl-pragma-256 inputs/gpu-images/$i $size $size
done