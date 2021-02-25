# Video Cleaning

### Problem

The file `shuffled_video_360.mp4` is a corrupted video file. The frames have been shuffled, and some frames are not from the original video. The goal is to restore the original video. 

### My solution(s)

I used a classifier to remove corrupted frames, and a similarity-based algorithm to put the frames back into order.
