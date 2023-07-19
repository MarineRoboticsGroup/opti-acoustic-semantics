#!/usr/bin/env bash

for file in /home/singhk/data/building_1_pool/6-27-23-testing/images/*.jpg
do
    name="$(basename "$file" .jpg)"
    mkdir -p "/home/singhk/data/building_1_pool/6-27-23-testing/images_dino_format/$name"
    mv $file "/home/singhk/data/building_1_pool/6-27-23-testing/images_dino_format/$name/$name.jpg"
done
