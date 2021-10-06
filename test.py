import av
from utils.ioutils import av_wrappers
from utils.nfov import NFOV
import numpy as np
import soundfile as sf
from utils.ambisonics.position import Position
from utils.video360 import rotate_ambix
from utils.video360 import er2polar
from utils.video360 import beamformer
import os
# here set the center point, if switch order or not, and audio file (three files are provided)
switch_order = True # True or False
center = [0.25, 0.5] # [0.25,0.5] or [0.75,0.5]
#audio_fn = 'data/yt360/audio/y6MBGQW7-sg-150.m4a'
audio_fn = 'data/yt360/audio/redownloaded_from_YT.m4a'
#audio_fn = 'data/yt360/audio/kqAZkf5Odyw-80.m4a'


# the ambisonics downloaded from youtube has 6 channles with 2 channles are zeros. The order is XYZ0W0
audio_ctr = av.open(audio_fn)

def load_audio_clip(ctr, start_time, duration):

    audio_dt, audio_fps = av_wrappers.av_audio_loader(container=ctr,rate=24000,start_time=start_time,duration=duration,layout='4.0')

    return audio_dt, audio_fps

st,dur=8.0,10.0

audio_dt, audio_fps = load_audio_clip(audio_ctr, st,dur)

# after loading it using av_audio_loader, the channels order turned into XYWZ, in order to make it to the right order(WYZX), we need to switch the oder mannully as follows,
if switch_order:
    audio_dt= audio_dt[[2,1,3,0],:]

pos_ambix = Position(*er2polar(*center), 'polar')
print('the polar coordinates are', 'phi is',er2polar(*center)[0],'nu is', er2polar(*center)[1] )
r_audio = rotate_ambix(audio_dt, pos_ambix)[0].astype(np.float32)

# to verify if the ambisonics are correctly rotated, we use beamformer to extract the signle channle out and one can easily listen to it.
center_1 = [0.5, 0.5]
ro_bf_audio = beamformer(r_audio,er2polar(*center_1))[0].astype(np.float32)

# save the data
if switch_order:
    save_data ='./positive_samples/'
else:
    save_data = './negative_samples/'
if not os.path.exists(save_data):
    os.makedirs(save_data)
    print("Directory " , save_data ,  " Created ")
else:
    print("Directory " , save_data ,  " already exists")

sf.write(save_data+str(center[0])+'_'+str(center[1])+'.wav',ro_bf_audio,24000)

# bf_audio = beamformer(audio_dt,er2polar(*center))[0].astype(np.float32)
# sf.write('bf_'+ str(center[0])+'_'+str(center[1])+'.wav',bf_audio,24000)


audio_ctr.close()
