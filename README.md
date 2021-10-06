# YT-360 ambisonics order

## This repo is to demonstrate the correct ambisonics order of YT-360 dataset

### To run this code,

```
python test.py
```

```
"switch_order" can be set True or False
"center point" can be set [0.25,0.5] or [0.75,0.5]
```
if center point is set to [0.25,0.5], which means azimuth angle is 90 degree and elevation angle is 0. If center point is [0.75,0.5] , which is azimuth angle is -90 degree and elevation angle is 0.

### Data provided,
Two video and their correponding audio data are provided. "kqAZkf5Odyw-80.mp4/m4a" is from the YT-360 dataset and "redownloaded_from_YT.mp4/m4a" is the one we made on our own. 

### Positive samples analysis,
if the order is switched, "0.25_0.5.wav" one can only hear piano and seagull sound, "0.75_0.5.wav" one can only hear voice and seagull sound, which is in line with the visual information.

### Negative samples analysis,
if the order is not switched manually, from both "0.25_0.5.wav" and "0.75_0.5.wav", one can hear every sound in it, voice, piano and seagull. 

### Simally analysis results can also be found in the other example. 
Since example "redownloaded_from_YT.mp4/m4a" is created in a way that strictly piano on left side, and voice on the right side, and seagull on the top side, after applying beamforming, if the order is appllied correctly, one can hear the difference easily. 

### acknowledgement
data "redownloaded_from_YT.mp4/m4a" is created by Archontis Politis and "kqAZkf5Odyw-80.mp4/m4a" is from theclarksmusic https://www.youtube.com/watch?v=kqAZkf5Odyw
