ffmpeg -i raw/out.mp4 -vf crop=640:192:0:225 -ss 00:00:15 -t 00:20:00 data.mp4 
ffmpeg -i data.mp4 -vf fps=1/30 background/bg_%02d.png
