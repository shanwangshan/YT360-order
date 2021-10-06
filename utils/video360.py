import os
import math
import numpy as np
from utils.ambisonics.common import spherical_harmonics_matrix
from IPython import embed

def overlay_map(audio_fn, video_fn, output_fn):
    from utils.ioutils.video import VideoReader, VideoWriter
    from utils.ioutils.audio import AudioReader
    from utils.ambisonics.distance import SphericalAmbisonicsVisualizer
    import tempfile
    from matplotlib import pyplot as plt
    from skimage.transform import resize

    print('Overlaying spherical map')
    tmp_vid_file = tempfile.mktemp(dir='/tmp/', suffix='.avi')

    areader = AudioReader(audio_fn)
    vreader = VideoReader(video_fn)
    vgen = vreader._read()
    vwriter = VideoWriter(tmp_vid_file, vreader._fps, vreader.width, vreader.height)
    cmap = plt.cm.YlOrRd(np.linspace(0, 1, 256))[:, :3]

    snd_fps = 5000
    ambix = areader.read(target_fps=snd_fps)
    ambiVis = SphericalAmbisonicsVisualizer(ambix, snd_fps, 5. / vreader._fps, 5.)

    cur_rms = ambiVis.get_next_frame()
    cur_rms = (cur_rms - cur_rms.min()) / (cur_rms.max() - cur_rms.min() + 0.005)
    while True:
        prev_rms = cur_rms
        cur_rms = ambiVis.get_next_frame()
        if cur_rms is None:
            break
        cur_rms = (cur_rms - cur_rms.min()) / (cur_rms.max() - cur_rms.min() + 0.005)

        for i in range(5):
            frame = next(vgen)
            if frame is None:
                break

            # Create map
            beta = i / 5.
            rms = (1 - beta) * prev_rms + beta * cur_rms
            rms = rms * 2. - 0.7
            rms[rms < 0] = 0
            dir_map = (rms * 255).astype(int)
            dir_map[dir_map > 255] = 255
            dir_map = resize(cmap[dir_map], (vreader.height, vreader.width)) * 255

            alpha = resize(rms[:, :, np.newaxis], (vreader.height, vreader.width)) * 0.6
            overlay = alpha * dir_map + (1 - alpha) * frame
            vwriter.write_frame(overlay.astype(np.uint8))

    cmd = 'ffmpeg -y -i {} -i {} -vcodec copy -strict -2 {}'.format(audio_fn, tmp_vid_file, output_fn)
    os.system(cmd)
    os.remove(tmp_vid_file)


def er2polar(x, y, r=1):
    '''
    map position in equirectangular map to polar coordinates
    '''


    phi = (1 - 2 * x) * math.pi
    nu = (1 - 2 * y) * math.pi / 2
    return phi, nu, r


class AmbiPower:
    def __init__(self, pos):
        from utils.ambisonics.decoder import AmbiDecoder
        from utils.ambisonics.position import Position

        pos = [Position(*er2polar(x, y), 'polar') for x, y in pos]
        self.decoder = AmbiDecoder(pos, method='projection')

    def compute(self, ambi):
        sig = self.decoder.decode(ambi)
        sig_pow = np.sum(sig ** 2, axis=1)
        return sig_pow


def ambix_power_map(ambix, audio_rate=22050, outp_rate=10, angular_res=5.0):
    from utils.ambisonics.distance import spherical_mesh
    from utils.ambisonics.decoder import AmbiDecoder, AmbiFormat
    from utils.ambisonics.position import Position
    phi_mesh, nu_mesh = spherical_mesh(angular_res)
    mesh_p = [Position(phi, nu, 1., 'polar') for phi, nu in zip(phi_mesh.reshape(-1), nu_mesh.reshape(-1))]
    ambi_order = math.sqrt(ambix.shape[0]) - 1
    decoder = AmbiDecoder(mesh_p, AmbiFormat(ambi_order=int(ambi_order), sample_rate=audio_rate), method='projection')

    # Compute RMS at each speaker
    rms = []
    window_size = int(audio_rate / outp_rate)
    for t in np.arange(0, ambix.shape[1], window_size):
        chunk = ambix[:, int(t):int(t) + window_size]
        decoded = decoder.decode(chunk)
        rms += [np.flipud(np.sqrt(np.mean(decoded ** 2, 1)).reshape(phi_mesh.shape))]
    return np.stack(rms, 0)


def ambix_power_map_freq(audio, audio_rate, freq_lims):
    import librosa

    # Filter out frequencies
    audio_masked = []
    for a in audio:
        spect = librosa.core.stft(a)

        mask = (librosa.fft_frequencies(audio_rate) > freq_lims[0]).astype(float) * \
               (librosa.fft_frequencies(audio_rate) < freq_lims[1]).astype(float)
        spec_masked = spect * mask[:, np.newaxis]
        audio_masked += [librosa.core.istft(spec_masked)]
    audio_masked = np.stack(audio_masked, 0)

    # Compute source map
    audio_maps = ambix_power_map(audio_masked, audio_rate=audio_rate, outp_rate=5, angular_res=5.)
    return audio_maps


def rotate_ambix(ambi, pos):
    '''
    generate ambisonics aligned at pos
    '''
    from utils.ambisonics.position import Position
    from scipy.spatial.transform import Rotation as R

    if isinstance(pos, Position):
        pos = [pos]
    assert ambi.shape[0] == 4, ValueError('Only implemented for first order ambisonics (ACN format).')

    #ambi = ambi[[2,1,3,0],:]
    ambi_w = ambi[:1]
    ambi_xyz = ambi[[3, 1, 2]]
    out_ambi = []
    for p in pos:
        # http://www.matthiaskronlachner.com/wp-content/uploads/2013/01/Kronlachner_Master_Spatial_Transformations_Mobile.pdf
        r_yaw = R.from_euler('z', -p.phi).as_matrix()
        r_pitch = R.from_euler('y', p.nu).as_matrix()
        r_ambi = np.concatenate([ambi_w, (r_yaw @ r_pitch @ ambi_xyz)[[1, 2, 0]]])
        # r_ambi = np.concatenate([ambi_w, (r_pitch @ r_yaw @ ambi_xyz)[[1, 2, 0]]])
        out_ambi.append(r_ambi)

    return out_ambi


def beamformer(foasig,beam_dirs):

#phi, theta: azimuth, elevation coordinates in RHS

# Assuming ACN/SN3D convention
#   ch1: omni
#   ch2: dipole-y
#   ch3: dipole-z
#   ch4: dipole-x

# foasig: the FOA recording [lSigx4 array]
# beam_dirs: the beamforming directions [K x 2 array]
#            [azi1 elev1; ...; azi_K elev_K] in rads


    #foasig = foasig[[2,1,3,0],:]
    foasig = foasig.T # [4,24000] to [24000,4]

    #lSig = size(foasig,1)
    lSig = np.shape(foasig)[0] # 24000

    # foasig = np.array([[1,4,7,8],[2,4,5,7],[5,4,6,8]])
    # beam_dirs = (np.pi/4,np.pi/3,1)

    beam_dirs = beam_dirs[:2]
    beam_dirs = np.asarray([beam_dirs])
    #nBeams = size(beam_dirs,1)
    nBeams = np.shape(beam_dirs)[0]
    # convert spherical coords to cartesian
    beam_xyz = unitSph2cart(beam_dirs)
    # form beamweights for ACN/SN3D

    beam_weights = beam_xyz[:,[1, 2, 0]] #beam_xyz[:,[0, 1, 2]] #
    # beam_weights = [0.5*ones(nBeams,1) 0.5*beam_weights] # % setup for cardioid beam
    #__import__("pdb").set_trace()

    beam_weights =np.hstack((0.5*np.ones((nBeams,1)), 0.5*beam_weights)) # % setup for cardioid beam

    # # apply beamforming weights to get lSig x K beamfomed signals
    bfsig = foasig @ beam_weights.T

    return bfsig.T
def unitSph2cart(aziElev):
#UNITSPH2CART Get coordinates of unit vectors from azimuth-elevation
#   Similar to sph2cart, assuming unit vectors and using matrices for the
#   inputs and outputs, instead of separate vectors of coordinates.
    #if size(aziElev,2)~=2, aziElev = aziElev.T end
    xyz=[]
    #import pdb; pdb.set_trace()
    x = np.cos(aziElev[:,0]) * np.cos(aziElev[:,1])
    x = x[0]
    y = np.sin(aziElev[:,0]) * np.cos(aziElev[:,1])
    y = y[0]
    z = np.sin(aziElev[:,1])
    z= z[0]
    xyz.append((x,y,z))
    xyz = np.array((xyz))

    return xyz


def rotate_video(video_fn, output_fn, pos):
    from utils.ioutils.video import VideoReader, VideoWriter
    vreader = VideoReader(video_fn)
    vwriter = VideoWriter(output_fn, vreader._fps, vreader.width, vreader.height)
    h_shift_pixels = int((np.mod(pos.phi + np.pi, 2*np.pi) - np.pi) / np.pi * vreader.width) // 2
    v_shift_pixels = int((np.mod(pos.nu + np.pi/2., np.pi) - np.pi/2.) / (np.pi/2) * vreader.height) // 2
    for frame in vreader.read(0, 5.):
        frame = np.array(frame)
        frame = np.roll(frame, shift=h_shift_pixels, axis=1)
        frame = np.roll(frame, shift=v_shift_pixels, axis=0)
        vwriter.write_frame(frame.astype(np.uint8))


def project_audio(ambix, center):
    order = math.sqrt(ambix.shape[0])-1
    Y = spherical_harmonics_matrix([center], order)
    crop = Y @ ambix
    return crop


def audio_crop_freq_sep(ambix, center, sigma=9, thr=0.7):
    import librosa
    from scipy import ndimage
    center_stft = librosa.core.stft(project_audio(ambix, center=center)[0])
    center_stft_smooth = ndimage.gaussian_filter(np.abs(center_stft), sigma=sigma)

    other_stft_smooth = []
    for phi in np.linspace(-pi, pi, 16):
        for nu in np.linspace(-pi / 2, pi / 2, 8):
            p = Position(phi, nu, 1., 'polar')
            if (p.coords('cartesian') * center.coords('cartesian')).sum() > math.cos(pi / 2):
                continue
            stft = librosa.core.stft(project_audio(ambix, center=p)[0])
            other_stft_smooth += [ndimage.gaussian_filter(np.abs(stft), sigma=sigma)]

    other_stft_smooth = np.stack(other_stft_smooth, 0)
    rank = (np.abs(center_stft_smooth[np.newaxis]) > np.abs(other_stft_smooth)).sum(0) / other_stft_smooth.shape[0]

    stft = librosa.core.stft(project_audio(ambix, center=center)[0])
    stft[rank < thr] = 0
    wav = librosa.core.istft(stft)
    return wav


def audio_binary_spatial_separation(ambix, pos1, pos2, sigma=9):
    import librosa
    from scipy import ndimage

    pos1_stft = librosa.core.stft(project_audio(ambix, center=pos1)[0])
    pos1_stft_smooth = ndimage.gaussian_filter(np.abs(pos1_stft), sigma=sigma)

    pos2_stft = librosa.core.stft(project_audio(ambix, center=pos2)[0])
    pos2_stft_smooth = ndimage.gaussian_filter(np.abs(pos2_stft), sigma=sigma)

    mask = (pos1_stft_smooth > pos2_stft_smooth).astype(float)
    pos1_wav = librosa.core.istft(pos1_stft * mask)
    pos2_wav = librosa.core.istft(pos2_stft * (1. - mask))
    return pos1_wav, pos2_wav


if __name__ == '__main__':
    from utils.ioutils.audio import AudioReader, AudioWriter
    from utils.ambisonics.position import Position
    from math import pi

    yid = 'bb5eETSspVI-213'
    # yid = 'l5M8AvP6rvs-518'

    for rot_theta in [pi/2.]:
        for rot_pitch in [pi/6.]:
            audio_fn = f'rot-yaw{rot_theta:.3f}-pitch{rot_pitch:.3f}.m4a'
            video_fn = f'rot-yaw{rot_theta:.3f}-pitch{rot_pitch:.3f}.mp4'
            out_fn = f'out-yaw{rot_theta:.3f}-pitch{rot_pitch:.3f}.mp4'
            pos = Position(rot_theta, rot_pitch, 1., c_type='polar')
            ambix = AudioReader(f'data/spatial-audio-db/audio/{yid}.m4a').read(0., 5., 24000)
            ambix_rot = rotate_ambix(ambix, pos)[0]

            AudioWriter(audio_fn, 24000).write(ambix_rot)
            rotate_video(f'data/spatial-audio-db/video/{yid}.mp4', video_fn, pos)
            overlay_map(audio_fn, video_fn, out_fn)
