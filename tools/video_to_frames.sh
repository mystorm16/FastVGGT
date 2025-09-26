#!/bin/bash

# Check if correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_video.mp4> <output_folder> <fps>"
    exit 1
fi

input_video="$1"
output_folder="$2"
fps="$3"

# Check if input video exists
if [ ! -f "$input_video" ]; then
    echo "Error: Input video file does not exist"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$output_folder"

# Extract frames using ffmpeg
ffmpeg -i "$input_video" \
       -vf "fps=$fps" \
       -frame_pts 1 \
       -f image2 \
       -quality 2 \
       "${output_folder}/%06d.jpg"

echo "Frames extracted successfully to $output_folder"