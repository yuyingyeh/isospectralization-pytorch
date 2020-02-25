ffmpeg -v 0 -pattern_type glob -i "$1" -vf palettegen -y palette.png
ffmpeg -v 0 -r 1 -pattern_type glob -i "$1" -i palette.png -lavfi paletteuse -y "$2"
rm palette.png
