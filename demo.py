# Animation Controllers
# Amplify the lip motion in horizontal direction
AMP_LIP_SHAPE_X = 2  # min:0.5, max:5.0

# Amplify the lip motion in vertical direction
AMP_LIP_SHAPE_Y = 2  # min:0.5, max:5.0

# Amplify the head pose motion (usually smaller than 1.0, put it to 0. for a static head pose)
AMP_HEAD_POSE_MOTION = 2  # min:0.0, max:1.0

# Add naive eye blink
ADD_NAIVE_EYE = True  # ["False", "True"]

# If your image has an opened mouth, put this as True, else False
CLOSE_INPUT_FACE_MOUTH = False  # ["False", "True"]

# Landmark Adjustment
# Adjust upper lip thickness (postive value means thicker)
UPPER_LIP_ADJUST = -1  # min:-3.0, max:3.0, step:1.0}

# Adjust lower lip thickness (postive value means thicker)
LOWER_LIP_ADJUST = -1  # min:-3.0, max:3.0, step:1.0}

# Adjust static lip width (in multipication)
LIP_WIDTH_ADJUST = 1.0  # min:0.8, max:1.2, step:0.01}

import sys

sys.path.append("thirdparty/AdaptiveWingLoss")
import os, glob
import numpy as np
import cv2
import argparse
# from src.approaches.train_image_translation import Image_translation_block
# import torch
import pickle
import face_alignment
from src.autovc.AutoVC_mel_Convertor_retrain_version import AutoVC_mel_Convertor
import shutil
# import time
import util.utils as util
# from scipy.signal import savgol_filter
from src.approaches.train_audio2landmark import Audio2landmark_model

# import ctypes
#
# hllDll = ctypes.WinDLL(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/bin/cublas64_11.dll")

# sys.stdout = open(os.devnull, 'a')

parser = argparse.ArgumentParser()
parser.add_argument('--jpg', type=str, default='obama.jpg')
parser.add_argument('--close_input_face_mouth', default=CLOSE_INPUT_FACE_MOUTH, action='store_true')
parser.add_argument('--load_AUTOVC_name', type=str, default='examples/ckpt/ckpt_autovc.pth')
parser.add_argument('--load_a2l_G_name', type=str, default='examples/ckpt/ckpt_speaker_branch.pth')
parser.add_argument('--load_a2l_C_name', type=str,
                    default='examples/ckpt/ckpt_content_branch.pth')  # ckpt_audio2landmark_c.pth')
parser.add_argument('--load_G_name', type=str,
                    default='examples/ckpt/ckpt_116_i2i_comb.pth')  # ckpt_image2image.pth') #ckpt_i2i_finetune_150.pth') #c
parser.add_argument('--amp_lip_x', type=float, default=AMP_LIP_SHAPE_X)
parser.add_argument('--amp_lip_y', type=float, default=AMP_LIP_SHAPE_Y)
parser.add_argument('--amp_pos', type=float, default=AMP_HEAD_POSE_MOTION)
parser.add_argument('--reuse_train_emb_list', type=str, nargs='+',
                    default=[])  # ['iWeklsXc0H8']) #['45hn7-LXDX8']) #['E_kmpT-EfOg']) #'iWeklsXc0H8', '29k8RtSUjE0', '45hn7-LXDX8',
parser.add_argument('--add_audio_in', default=False, action='store_true')
parser.add_argument('--comb_fan_awing', default=False, action='store_true')
parser.add_argument('--output_folder', type=str, default='output')
parser.add_argument('--test_end2end', default=True, action='store_true')
parser.add_argument('--dump_dir', type=str, default='', help='')
parser.add_argument('--pos_dim', default=7, type=int)
parser.add_argument('--use_prior_net', default=True, action='store_true')
parser.add_argument('--transformer_d_model', default=32, type=int)
parser.add_argument('--transformer_N', default=2, type=int)
parser.add_argument('--transformer_heads', default=2, type=int)
parser.add_argument('--spk_emb_enc_size', default=16, type=int)
parser.add_argument('--init_content_encoder', type=str, default='')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--reg_lr', type=float, default=1e-6, help='weight decay')
parser.add_argument('--write', default=False, action='store_true')
parser.add_argument('--segment_batch_size', type=int, default=1, help='batch size')
parser.add_argument('--emb_coef', default=3.0, type=float)
parser.add_argument('--lambda_laplacian_smooth_loss', default=1.0, type=float)
parser.add_argument('--use_11spk_only', default=False, action='store_true')
parser.add_argument('-f')
opt_parser = parser.parse_args()

# img = cv2.imread('examples/obama.jpg')
# predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu', flip_input=True)
# shapes = predictor.get_landmarks(img)
# if not shapes or len(shapes) != 1:
#     print('Cannot detect face landmarks. Exit.')
#     exit(-1)
#
shape_3d = np.loadtxt('landmarks/obama.txt')
if opt_parser.close_input_face_mouth:
    util.close_input_face_mouth(shape_3d)
shape_3d[48:, 0] = (shape_3d[48:, 0] - np.mean(shape_3d[48:, 0])) * LIP_WIDTH_ADJUST + np.mean(
    shape_3d[48:, 0])  # wider lips
shape_3d[49:54, 1] -= UPPER_LIP_ADJUST  # thinner upper lip
shape_3d[55:60, 1] += LOWER_LIP_ADJUST  # thinner lower lip
shape_3d[[37, 38, 43, 44], 1] -= 2.  # larger eyes
shape_3d[[40, 41, 46, 47], 1] += 2.  # larger eyes
shape_3d, scale, shift = util.norm_input_face(shape_3d)

au_data = []
au_emb = []
ains = glob.glob1('input', '*.wav')
ains = [item for item in ains if item != 'tmp.wav']
ains.sort()
print(len(ains))
for ain in ains:
    # os.system('ffmpeg -y -loglevel error -i examples/{} -ar 16000 examples/tmp.wav'.format(ain))
    # shutil.copyfile('examples/tmp.wav', 'examples/{}'.format(ain))

    # au embedding
    from thirdparty.resemblyer_util.speaker_emb import get_spk_emb

    me, ae = get_spk_emb('input/{}'.format(ain))
    au_emb.append(me.reshape(-1))

    print('Processing audio file', ain)
    c = AutoVC_mel_Convertor('examples')

    au_data_i = c.convert_single_wav_to_autovc_input(audio_filename=os.path.join('input', ain),
                                                     autovc_model_path=opt_parser.load_AUTOVC_name)
    au_data += au_data_i
# if(os.path.isfile('examples/tmp.wav')):
#     os.remove('examples/tmp.wav')

print("Loaded audio...", file=sys.stderr)

# landmark fake placeholder
fl_data = []
rot_tran, rot_quat, anchor_t_shape = [], [], []
for au, info in au_data:
    au_length = au.shape[0]
    fl = np.zeros(shape=(au_length, 68 * 3))
    fl_data.append((fl, info))
    rot_tran.append(np.zeros(shape=(au_length, 3, 4)))
    rot_quat.append(np.zeros(shape=(au_length, 4)))
    anchor_t_shape.append(np.zeros(shape=(au_length, 68 * 3)))

print(f'{len(fl_data)} {len(au_data)}')

if os.path.exists(os.path.join('examples', 'dump', 'random_val_fl.pickle')):
    os.remove(os.path.join('examples', 'dump', 'random_val_fl.pickle'))
if os.path.exists(os.path.join('examples', 'dump', 'random_val_fl_interp.pickle')):
    os.remove(os.path.join('examples', 'dump', 'random_val_fl_interp.pickle'))
if os.path.exists(os.path.join('examples', 'dump', 'random_val_au.pickle')):
    os.remove(os.path.join('examples', 'dump', 'random_val_au.pickle'))
if os.path.exists(os.path.join('examples', 'dump', 'random_val_gaze.pickle')):
    os.remove(os.path.join('examples', 'dump', 'random_val_gaze.pickle'))

with open(os.path.join('examples', 'dump', 'random_val_fl.pickle'), 'wb') as fp:
    pickle.dump(fl_data, fp)
with open(os.path.join('examples', 'dump', 'random_val_au.pickle'), 'wb') as fp:
    pickle.dump(au_data, fp)
with open(os.path.join('examples', 'dump', 'random_val_gaze.pickle'), 'wb') as fp:
    gaze = {'rot_trans': rot_tran, 'rot_quat': rot_quat, 'anchor_t_shape': anchor_t_shape}
    pickle.dump(gaze, fp)

model = Audio2landmark_model(opt_parser, jpg_shape=shape_3d)
if len(opt_parser.reuse_train_emb_list) == 0:
    model.test(au_emb=au_emb)
else:
    model.test(au_emb=None)

# print("Audio->Landmark...", file=sys.stderr)

# fls = glob.glob1('examples', 'pred_fls_*.txt')
# fls.sort()
