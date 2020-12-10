import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from time import time
import argparse

from api import PRN
from utils.write import write_obj_with_colors


def frames_to_objs(frames, save_folder, name = ""):
  for idx in range(frames.shape[0]):
    out_obj = os.path.join(save_folder, name + "_" + str(idx) + ".obj")
    out_mat = os.path.join(save_folder, name + "_" + str(idx) + "_mesh.mat")
    frame = frames[idx, :, :, :]
    pos = prn.process(frame)

    """ get landmarks --> Saved to a txt file (not useful for us)"""
    kpt = prn.get_landmarks(pos)
    """ 3D vertices"""
    vertices = prn.get_vertices(pos)
    """ Corresponding colors """
    colors = prn.get_colors(frame, vertices)
    print(colors)
    print(colors.shape)

    write_obj_with_colors(out_obj, vertices, prn.triangles, colors)
    # np.savetxt(os.path.join(save_folder, name + "_" + str(idx) + '.txt'), kpt) 
    
    print("Outputted {}".format(idx))
    # sio.savemat(out_mat, {'vertices': vertices, 'colors': colors, 'triangles': prn.triangles})



  
# ---- init PRN
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # GPU number, -1 for CPU
prn = PRN(is_dlib = True)

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')
parser.add_argument('--frames_path', type=str, help='NPath to .npy file', required=True)
parser.add_argument('--out_folder', type=str, help='Folder for output', required=True)
parser.add_argument('--out_name', type=str, help='Name for output', required=False
args = parser.parse_args()

print(args.frames_path)
print(args.out_folder)
print(args.out_name)

frames = np.load(args.frames_path)
print(frames.shape)
frames_to_objs(frames, args.out_folder, name=args.out_name)