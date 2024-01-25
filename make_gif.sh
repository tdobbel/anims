#!/bin/bash

lossy=80

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -r|--frame-rate) frame_rate=$2; shift;;
        -f|--frame-format) frame_format="$2"; shift;;
        -o|--output-name) output_name="$2"; shift;;
        -l|--lossy) lossy="$2"; shift;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

ffmpeg -y -r $frame_rate -f image2 -vb 5000k -i anim/frame_%04d.png -vf "scale=trunc(iw/4)*2:trunc(ih/4)*2" -pix_fmt yuv420p $output_name.mp4
ffmpeg -y -i $output_name.mp4 -filter_complex "[0:v] palettegen" palette.png
ffmpeg -y -i $output_name.mp4 -i palette.png -filter_complex "[0:v][1:v] paletteuse" $output_name.gif
optimized=$(printf "%s_optimized.gif" $output_name)
gifsicle -O3 --lossy=$lossy --colors 256 $output_name.gif -o $optimized
