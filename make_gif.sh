frame_rate=6
anim_name=anim_qaruh_streamlines


ffmpeg -y -r $frame_rate -f image2 -vb 5000k -i anim/frame_%04d.png -vf "scale=trunc(iw/4)*2:trunc(ih/4)*2" -pix_fmt yuv420p $anim_name.mp4
ffmpeg -y -i $anim_name.mp4 -filter_complex "[0:v] palettegen" palette.png
ffmpeg -y -i $anim_name.mp4 -i palette.png -filter_complex "[0:v][1:v] paletteuse" $anim_name.gif
optimized=$(printf "%s_optimized.gif" $anim_name)
gifsicle -O3 --lossy=80 --colors 256 $anim_name.gif -o $optimized

