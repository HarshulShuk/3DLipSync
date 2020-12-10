# Stereoscopic Approach (CycleGAN + Wav2Lip)
Kushagra Pundeer
Relevant code for 670 research paper

References: 
[1]  https://github.com/andrea-pilzer/unsup-stereo-depthGAN
[2]  https://github.com/Rudrabha/Wav2Lip
[3]  https://github.com/rickgroen/depthgan
[4]  https://github.com/alwynmathew/bilinear-sampler-pytorch

Added or Modified files. 

1. Added Stereoscopic_Approach/main_cycle_wav2lip_left.py file for combined model with left stereo image dataset (Wav2Lip) & Left-Right Stereo for CycleGAN | [2] and [3] used as reference

2. Added Stereoscopic_Approach/main_cycle_wav2lip_right.py file for combined model with right stereo image dataset (Wav2Lip) & Left-Right Stereo for CycleGAN | [2] and [3] used as reference

3. Helper Files for data-collection, pre-processing etc capture_frames.py Extract_left_right.py and Stereo_dataset_split.py

4. Some modifications made that are commented to reference code files networks/utils.py networks/generator_resnet.py data_loader/transforms.py config_parameters.py architectures/monodepth_architecture.py losses/monodepth_loss.py

main_cycle_wav2lip_left.py, main_cycle_wav2lip_right.py : Wav2Lip Trained on Left and Right Stereo pairs individually, CycleGAN trains on stereo-pair dataset. 


# PRNet Approach
Harshul Shukla
Relevant code for CS670 research paper. 



## PRNet_scripts
These scripts directly use code from or make refrences to files in [PRNet](https://github.com/YadiraF/PRNet)
- **run_modified.py**: Script used to run PRNet given frames from a film as numpy file and then output each as .obj file. Uses other parts of PRNet library. Used a few lines from PRNet's [run_basics.py](https://github.com/YadiraF/PRNet/blob/master/run_basics.py)
- **train2.py**: Uses parts of api.py, predictor.py that defined network structure and pre processing that is used when loading images. Loaded image of new weights for modified MSE. No training example file was provided in PRNet, so this file mimicks the structure of the network, loads the weights, and trains. Saved weights used w/ run_modified.py afterwards to produce .objs
- **makemask.py**: Simple script to edit the weight mask




## helper_scripts
- **300W.py** : Used to generate ground truth for 300W dataset. Copied heavily from examples section in Face3D (Only main function is changed) which was also used for processing. See [8_generate_posmap_300WLP.py](https://github.com/YadiraF/face3d/blob/master/examples/8_generate_posmap_300WLP.py). Also followed instructions found [here](https://github.com/YadiraF/face3d/blob/master/examples/Data/BFM/readme.md) and ran generate.m to prepare BFM data used in 300W.py
- **makenp.py** : Used to turn a movie -> numpy file to be used in PRNet's **run_modified.py**[StackOverflow Ref](https://stackoverflow.com/questions/42163058/how-to-turn-a-video-into-numpy-array)

