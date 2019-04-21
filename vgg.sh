#!/usr/bin/env bash

python data_vgg.py data

for i in $(ls npy_data); do
    echo $i;
    python image_to_vector.py $i;
done;

python final_vgg.py