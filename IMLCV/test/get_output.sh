#!/bin/bash


echo "Bash version ${BASH_VERSION}..."

mkdir -p output/$1/output

# module load ImageMagick

for VARIABLE in $(seq 0 $2)
do
    echo "copying $VARIABLE"
    cp output/$1/round_$VARIABLE/combined.pdf output/$1/output/combined_$VARIABLE.pdf
    cp output/$1/FES_thermolib_$VARIABLE.pdf output/$1/output/FES_$VARIABLE.pdf

    echo "converting $VARIABLE"
    convert -density 200 output/$1/output/combined_$VARIABLE.pdf output/$1/output/combined_$VARIABLE.png
    convert -density 200 output/$1/output/FES_$VARIABLE.pdf output/$1/output/FES_$VARIABLE.png
    echo "merging $VARIABLE"
    convert +append output/$1/output/combined_$VARIABLE.png output/$1/output/FES_$VARIABLE.png output/$1/output/merged_$VARIABLE.png
done

ls output/$1/output


module load FFmpeg

echo "merged gif"
ffmpeg -r 1 -i output/$1/output/merged_%d.png -vf "scale=-1:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 output/$1/output/merged.gif
