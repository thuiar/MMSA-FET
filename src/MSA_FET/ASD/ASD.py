import glob
import math
import os
import subprocess
import warnings
from pathlib import Path
from shutil import move, rmtree

import cv2
import numpy
import python_speech_features
import torch
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.video_manager import VideoManager
from scipy import signal
from scipy.interpolate import interp1d
from scipy.io import wavfile

from .model import S3FD
from .talkNet import talkNet

warnings.filterwarnings("ignore")

def scene_detect(args):
    # CPU: Scene detection, output is the list of each shot's time duration
    videoManager = VideoManager([args['videoFilePath']])
    statsManager = StatsManager()
    sceneManager = SceneManager(statsManager)
    sceneManager.add_detector(ContentDetector())
    baseTimecode = videoManager.get_base_timecode()
    videoManager.set_downscale_factor()
    videoManager.start()
    sceneManager.detect_scenes(frame_source = videoManager, show_progress=False)
    sceneList = sceneManager.get_scene_list(baseTimecode)
    if sceneList == []:
        sceneList = [(videoManager.get_base_timecode(),videoManager.get_current_timecode())]
    return sceneList

def inference_video(args):
    # GPU: Face detection, output is the list contains the face location and score in this frame
    DET = S3FD(device='cuda')
    flist = glob.glob(os.path.join(args['pyframesPath'], '*.jpg'))
    flist.sort()
    dets = []
    for fidx, fname in enumerate(flist):
        image = cv2.imread(fname)
        imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = DET.detect_faces(imageNumpy, conf_th=0.9, scales=[args['facedetScale']])
        dets.append([])
        for bbox in bboxes:
          dets[-1].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]}) # dets has the frames info, bbox info, conf info
        # sys.stderr.write('%s-%05d; %d dets\r' % (args['videoFilePath'], fidx, len(dets[-1])))
    return dets

def bb_intersection_over_union(boxA, boxB, evalCol = False):
    # CPU: IOU Function to calculate overlap between two image
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if evalCol == True:
        iou = interArea / float(boxAArea)
    else:
        iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def track_shot(args, sceneFaces):
    # CPU: Face tracking
    iouThres  = 0.5     # Minimum IOU between consecutive face detections
    tracks    = []
    while True:
        track     = []
        for frameFaces in sceneFaces:
            for face in frameFaces:
                if track == []:
                    track.append(face)
                    frameFaces.remove(face)
                elif face['frame'] - track[-1]['frame'] <= args['numFailedDet']:
                    iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                    if iou > iouThres:
                        track.append(face)
                        frameFaces.remove(face)
                        continue
                else:
                    break
        if track == []:
            break
        elif len(track) > args['minTrack']:
            frameNum    = numpy.array([ f['frame'] for f in track ])
            bboxes      = numpy.array([numpy.array(f['bbox']) for f in track])
            frameI      = numpy.arange(frameNum[0],frameNum[-1]+1)
            bboxesI    = []
            for ij in range(0,4):
                interpfn  = interp1d(frameNum, bboxes[:,ij])
                bboxesI.append(interpfn(frameI))
            bboxesI  = numpy.stack(bboxesI, axis=1)
            if max(numpy.mean(bboxesI[:,2]-bboxesI[:,0]), numpy.mean(bboxesI[:,3]-bboxesI[:,1])) > args['minFaceSize']:
                tracks.append({'frame':frameI,'bbox':bboxesI})
    return tracks

def crop_video(args, track, cropFile):
    # CPU: crop the face clips
    flist = glob.glob(os.path.join(args['pyframesPath'], '*.jpg')) # Read the frames
    flist.sort()
    vOut = cv2.VideoWriter(cropFile + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (224,224))# Write video
    dets = {'x':[], 'y':[], 's':[]}
    for det in track['bbox']: # Read the tracks
        dets['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2) 
        dets['y'].append((det[1]+det[3])/2) # crop center x 
        dets['x'].append((det[0]+det[2])/2) # crop center y
    dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  # Smooth detections 
    dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
    dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
    for fidx, frame in enumerate(track['frame']):
        cs  = args['cropScale']
        bs  = dets['s'][fidx]   # Detection box size
        bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount 
        image = cv2.imread(flist[frame])
        frame = numpy.pad(image, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
        my  = dets['y'][fidx] + bsi  # BBox center Y
        mx  = dets['x'][fidx] + bsi  # BBox center X
        face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
        vOut.write(cv2.resize(face, (224, 224)))
    audioTmp    = cropFile + '.wav'
    audioStart  = (track['frame'][0]) / 25
    audioEnd    = (track['frame'][-1]+1) / 25
    vOut.release()
    command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -ss %.3f -to %.3f %s -loglevel panic" % \
              (args['audioFilePath'], audioStart, audioEnd, audioTmp)) 
    subprocess.call(command, shell=True, stdout=None) # Crop audio file
    # _, audio = wavfile.read(audioTmp)
    command = ("ffmpeg -y -i %st.avi -i %s -c:v copy -c:a copy %s.avi -loglevel panic" % \
              (cropFile, audioTmp, cropFile)) # Combine audio and video file
    subprocess.call(command, shell=True, stdout=None)
    os.remove(cropFile + 't.avi')
    return {'track':track, 'proc_track':dets}

def evaluate_network(files, args):
    # GPU: active speaker detection by pretrained TalkNet
    s = talkNet()
    weight_file = Path(__file__).parent.parent / 'exts' / 'pretrained' / 'TalkSet.pth'
    s.loadParameters(weight_file)
    s.eval()
    allScores = []
    # durationSet = {1,2,4,6} # To make the result more reliable
    durationSet = {1,1,1,2,2,2,3,3,4,5,6} # Use this line can get more reliable result
    for file in files:
        fileName = os.path.splitext(file.split('/')[-1])[0] # Load audio and video
        _, audio = wavfile.read(os.path.join(args['pycropPath'], fileName + '.wav'))
        audioFeature = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025, winstep = 0.010)
        video = cv2.VideoCapture(os.path.join(args['pycropPath'], fileName + '.avi'))
        videoFeature = []
        while video.isOpened():
            ret, frames = video.read()
            if ret == True:
                face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (224,224))
                face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
                videoFeature.append(face)
            else:
                break
        video.release()
        videoFeature = numpy.array(videoFeature)
        length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0])
        audioFeature = audioFeature[:int(round(length * 100)),:]
        videoFeature = videoFeature[:int(round(length * 25)),:,:]
        allScore = [] # Evaluation use TalkNet
        for duration in durationSet:
            batchSize = int(math.ceil(length / duration))
            scores = []
            with torch.no_grad():
                for i in range(batchSize):
                    inputA = torch.FloatTensor(audioFeature[i * duration * 100:(i+1) * duration * 100,:]).unsqueeze(0).cuda()
                    inputV = torch.FloatTensor(videoFeature[i * duration * 25: (i+1) * duration * 25,:,:]).unsqueeze(0).cuda()
                    embedA = s.model.forward_audio_frontend(inputA)
                    embedV = s.model.forward_visual_frontend(inputV)	
                    embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
                    out = s.model.forward_audio_visual_backend(embedA, embedV)
                    score = s.lossAV.forward(out, labels = None)
                    scores.extend(score)
            allScore.append(scores)
        allScore = numpy.round((numpy.mean(numpy.array(allScore), axis = 0)), 1).astype(float)
        allScores.append(allScore)	
    return allScores

def visualization(tracks, scores, args):
    # CPU: visulize the result for video format
    flist = glob.glob(os.path.join(args['pyframesPath'], '*.jpg'))
    flist.sort()
    faces = [[] for i in range(len(flist))]
    for tidx, track in enumerate(tracks):
        score = scores[tidx]
        for fidx, frame in enumerate(track['track']['frame'].tolist()):
            s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)] # average smoothing
            s = numpy.mean(s)
            faces[frame].append({'track':tidx, 'score':float(s),'s':track['proc_track']['s'][fidx], 'x':track['proc_track']['x'][fidx], 'y':track['proc_track']['y'][fidx]})
    talker_time = 0
    cs = args['cropScale']
    for fidx, fname in enumerate(flist):
        image = cv2.imread(fname)
        best_score, best_id = 0, -1
        for fid, face in enumerate(faces[fidx]):
            if face['score'] > best_score:
                best_score = face['score']
                best_id = fid
        if best_id != -1:
            face = faces[fidx][best_id]
            y1 = int(face['y']-face['s']*(1+cs))
            y2 = int(face['y']+face['s']*(1+2*cs))
            x1 = int(face['x']-face['s']*(1+cs))
            x2 = int(face['x']+face['s']*(1+cs))
            if y1 < 0:
                y1 = 0
            if y2 > image.shape[0]:
                y2 = image.shape[0]
            if x1 < 0:
                x1 = 0
            if x2 > image.shape[1]:
                x2 = image.shape[1]
            image_talk = image[y1:y2, x1:x2]
            temp_path = os.path.join(args['pytalkerPath'], str(talker_time).zfill(5) +'.jpg')
            cv2.imwrite(temp_path, image_talk)
            talker_time += 1

def run_ASD(input_video, output_dir, fps, args):

    assert Path(input_video).is_file(), f"{input_video} is not a file"

    # Initialization 
    args['pyaviPath'] = os.path.join(output_dir, 'pyavi')
    args['pyframesPath'] = os.path.join(output_dir, 'pyframes')
    args['pycropPath'] = os.path.join(output_dir, 'pycrop')
    args['pytalkerPath'] = os.path.join(output_dir, 'talker_face')
    os.makedirs(args['pyaviPath'], exist_ok = True) # The path for the input video, input audio, output video
    os.makedirs(args['pyframesPath'], exist_ok = True) # Save all the video frames
    os.makedirs(args['pycropPath'], exist_ok = True) # Save the detected face clips (audio+video) in this process
    os.makedirs(args['pytalkerPath'], exist_ok = True) # Save the detected face clips (audio+video) in this process

    # Extract video
    args['videoFilePath'] = os.path.join(args['pyaviPath'], 'video.avi')
    command = ("ffmpeg -y -i %s -qscale:v 2 -async 1 -r %d %s -loglevel panic" % \
        (input_video, fps, args['videoFilePath']))
    subprocess.call(command, shell=True, stdout=None)
    # sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the video and save in %s \r\n" %(args['videoFilePath']))
    
    # Extract audio
    args['audioFilePath'] = os.path.join(args['pyaviPath'], 'audio.wav')
    command = ("ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -ar 16000 %s -loglevel panic" % \
        (args['videoFilePath'], args['audioFilePath']))
    subprocess.call(command, shell=True, stdout=None)
    # sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the audio and save in %s \r\n" %(args['audioFilePath']))

    # Extract the video frames
    command = ("ffmpeg -y -i %s -qscale:v 2 -f image2 %s -loglevel panic" % \
        (args['videoFilePath'], os.path.join(args['pyframesPath'], '%06d.jpg'))) 
    subprocess.call(command, shell=True, stdout=None)
    # sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the frames and save in %s \r\n" %(args['pyframesPath']))

    # Scene detection for the video frames
    scene = scene_detect(args)

    # Face detection for the video frames
    faces = inference_video(args)

    # Face tracking
    allTracks, vidTracks = [], []
    for shot in scene:
        if shot[1].frame_num - shot[0].frame_num >= args['minTrack']: # Discard the shot frames less than minTrack frames
            allTracks.extend(track_shot(args, faces[shot[0].frame_num:shot[1].frame_num])) # 'frames' to present this tracks' timestep, 'bbox' presents the location of the faces
    # sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face track and detected %d tracks \r\n" %len(allTracks))

    # Face clips cropping
    for ii, track in enumerate(allTracks):
        vidTracks.append(crop_video(args, track, os.path.join(args['pycropPath'], '%05d'%ii)))
    # sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face Crop and saved in %s tracks \r\n" %args['pycropPath'])

    # Active Speaker Detection by TalkNet
    files = glob.glob("%s/*.avi"%args['pycropPath'])
    files.sort()
    scores = evaluate_network(files, args)

    # Visualization, save the result as the new video	
    visualization(vidTracks, scores, args)

    files = glob.glob(os.path.join(args['pytalkerPath'], '*.jpg'))
    for file in files:
        move(file, output_dir)
    rmtree(args['pyaviPath'])
    rmtree(args['pyframesPath'])
    rmtree(args['pycropPath'])
    rmtree(args['pytalkerPath'])


    