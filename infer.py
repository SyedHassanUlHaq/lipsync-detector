import os, pdb, glob, pickle, subprocess, cv2, numpy as np
from shutil import rmtree
from concurrent.futures import ThreadPoolExecutor
from scipy.interpolate import interp1d
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector
from detectors import S3FD
from SyncNetInstance import SyncNetInstance
from scipy import signal
from scipy.io import wavfile


SYNCNET_MODEL = None
FACE_DETECTOR = None

def load_models(device="cpu"):
    """
    Load SyncNet and face detector once at server startup.
    """
    global SYNCNET_MODEL, FACE_DETECTOR
    if SYNCNET_MODEL is None:
        SYNCNET_MODEL = SyncNetInstance()
        SYNCNET_MODEL.loadParameters("data/syncnet_v2.model")
    if FACE_DETECTOR is None:
        FACE_DETECTOR = S3FD(device=device)
    print("[INFO] Models loaded")

class Config:
    def __init__(self, video_path, reference, data_dir="data/work"):
        self.videofile = video_path
        self.reference = reference
        self.data_dir = data_dir

        self.frame_rate = 25
        self.facedet_scale = 0.25
        self.crop_scale = 0.4
        self.min_track = 100
        self.num_failed_det = 25
        self.min_face_size = 100
        self.batch_size = 20
        self.vshift = 15

        self.avi_dir = os.path.join(data_dir, "pyavi")
        self.tmp_dir = os.path.join(data_dir, "pytmp")
        self.work_dir = os.path.join(data_dir, "pywork")
        self.crop_dir = os.path.join(data_dir, "pycrop")
        self.frames_dir = os.path.join(data_dir, "pyframes")

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])

    return interArea/(boxAArea + boxBArea - interArea + 1e-6)

def scene_detect(cfg):
    video_manager = VideoManager([os.path.join(cfg.avi_dir, cfg.reference, "video.avi")])
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(ContentDetector())

    base_timecode = video_manager.get_base_timecode()
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    scene_list = scene_manager.get_scene_list(base_timecode)
    if not scene_list:
        scene_list = [(video_manager.get_base_timecode(), video_manager.get_current_timecode())]

    return scene_list

def detect_frame(DET, idx, fname, scale):
    if not os.path.isfile(fname):
        print(f"[WARN] Frame not found: {fname}")
        return idx, []
    
    image = cv2.imread(fname)
    if image is None:
        print(f"[WARN] Failed to read frame: {fname}")
        return idx, []
    
    image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bboxes = DET.detect_faces(image_np, conf_th=0.9, scales=[scale])
    return idx, bboxes

def inference_video(cfg):
    """
    Use preloaded FACE_DETECTOR instead of creating a new one every call.
    """
    DET = FACE_DETECTOR
    flist = sorted(glob.glob(os.path.join(cfg.frames_dir, cfg.reference, "*.jpg")))
    if not flist:
        raise FileNotFoundError(f"No frames found in {os.path.join(cfg.frames_dir, cfg.reference)}")
    dets = [[] for _ in flist]

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda f: detect_frame(DET, *f, cfg.facedet_scale), enumerate(flist)))

    for idx, bboxes in results:
        for bbox in bboxes:
            dets[idx].append({"frame": idx, "bbox": bbox[:-1].tolist(), "conf": bbox[-1]})

    return dets

def track_shot(cfg, scenefaces):
    iouThres = 0.5
    tracks = []

    faces_array = [list(f) for f in scenefaces]  # copy
    while True:
        track = []
        for framefaces in faces_array:
            for face in framefaces:
                if not track:
                    track.append(face)
                    framefaces.remove(face)
                elif face["frame"] - track[-1]["frame"] <= cfg.num_failed_det:
                    iou = bb_intersection_over_union(face["bbox"], track[-1]["bbox"])
                    if iou > iouThres:
                        track.append(face)
                        framefaces.remove(face)
                        continue
                else:
                    break
        if not track:
            break
        if len(track) > cfg.min_track:
            framenum = np.array([f["frame"] for f in track])
            bboxes = np.array([f["bbox"] for f in track])
            frame_i = np.arange(framenum[0], framenum[-1]+1)
            bboxes_i = np.stack([interp1d(framenum, bboxes[:, ij])(frame_i) for ij in range(4)], axis=1)
            if max(np.mean(bboxes_i[:, 2]-bboxes_i[:, 0]), np.mean(bboxes_i[:, 3]-bboxes_i[:, 1])) > cfg.min_face_size:
                tracks.append({"frame": frame_i, "bbox": bboxes_i})

    return tracks

        
def crop_video(opt,track,cropfile):

    flist = glob.glob(os.path.join(opt.frames_dir,opt.reference,'*.jpg'))
    flist.sort()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vOut = cv2.VideoWriter(cropfile+'t.avi', fourcc, opt.frame_rate, (224,224))

    dets = {'x':[], 'y':[], 's':[]}

    for det in track['bbox']:

        dets['s'].append(max((det[3]-det[1]),(det[2]-det[0]))/2) 
        dets['y'].append((det[1]+det[3])/2) 
        dets['x'].append((det[0]+det[2])/2) 

    dets['s'] = signal.medfilt(dets['s'],kernel_size=13)   
    dets['x'] = signal.medfilt(dets['x'],kernel_size=13)
    dets['y'] = signal.medfilt(dets['y'],kernel_size=13)

    for fidx, frame in enumerate(track['frame']):

        cs  = opt.crop_scale

        bs  = dets['s'][fidx]  
        bsi = int(bs*(1+2*cs)) 

        image = cv2.imread(flist[frame])
        
        frame = np.pad(image,((bsi,bsi),(bsi,bsi),(0,0)), 'constant', constant_values=(110,110))
        my  = dets['y'][fidx]+bsi  
        mx  = dets['x'][fidx]+bsi 

        face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
        
        vOut.write(cv2.resize(face,(224,224)))

    audiotmp    = os.path.join(opt.tmp_dir,opt.reference,'audio.wav')
    audiostart  = (track['frame'][0])/opt.frame_rate
    audioend    = (track['frame'][-1]+1)/opt.frame_rate

    vOut.release()

    command = ("ffmpeg -y -i %s -ss %.3f -to %.3f %s" % (os.path.join(opt.avi_dir,opt.reference,'audio.wav'),audiostart,audioend,audiotmp)) 
    output = subprocess.call(command, shell=True, stdout=None)

    if output != 0:
        pdb.set_trace()

    sample_rate, audio = wavfile.read(audiotmp)

    command = ("ffmpeg -y -i %st.avi -i %s -c:v copy -c:a copy %s.avi" % (cropfile,audiotmp,cropfile))
    output = subprocess.call(command, shell=True, stdout=None)

    if output != 0:
        pdb.set_trace()

    print('Written %s'%cropfile)

    os.remove(cropfile+'t.avi')

    print('Mean pos: x %.2f y %.2f s %.2f'%(np.mean(dets['x']),np.mean(dets['y']),np.mean(dets['s'])))

    return {'track':track, 'proc_track':dets}


def crop_faces(cfg, tracks):
    print("[INFO] Cropping faces...")

    crop_base = os.path.join(cfg.crop_dir, cfg.reference)
    os.makedirs(crop_base, exist_ok=True)

    for idx, track in enumerate(tracks):
        cropfile = os.path.join(crop_base, f"{idx:05d}")
        
        print(f"[INFO] Cropping track {idx} -> {cropfile}.avi")
        
        crop_video(cfg, track, cropfile)
        

def run_syncnet(cfg):
    s = SYNCNET_MODEL
    crop_path = os.path.join(cfg.crop_dir, cfg.reference)
    flist = sorted(glob.glob(os.path.join(crop_path, "0*.avi")))
    
    print(f"[INFO] Looking for crop files in: {crop_path}")
    print(f"[INFO] Found {len(flist)} crop files")
    
    if len(flist) == 0:
        print(f"[WARN] No crop files found for syncnet evaluation")
        return []
    
    dists = []
    for fname in flist:
        try:
            result = s.evaluate(cfg, videofile=fname)
            if result is not None and len(result) > 2:
                dist_array = result[2]
                dists.extend(np.array(dist_array).flatten().astype(float).tolist())
        except Exception as e:
            print(f"[WARN] Error evaluating syncnet for {fname}: {e}")
    
    print(f"[INFO] Computed {len(dists)} sync distances")
    return dists

def run_inference(video_path: str, reference: str, skip_persistent_save: bool = False):

    cfg = Config(video_path, reference)

    for folder in [cfg.work_dir, cfg.crop_dir, cfg.avi_dir, cfg.frames_dir, cfg.tmp_dir]:
        path = os.path.join(folder, reference)
        if os.path.exists(path):
            rmtree(path)
        os.makedirs(path)

    avi_file = os.path.join(cfg.avi_dir, reference, 'video.avi')
    frames_pattern = os.path.join(cfg.frames_dir, reference, '%06d.jpg')
    audio_file = os.path.join(cfg.avi_dir, reference, 'audio.wav')

    os.makedirs(os.path.dirname(frames_pattern), exist_ok=True)

    print(f"[INFO] Converting video to AVI format...")
    subprocess.call(f"ffmpeg -y -i {video_path} -qscale:v 2 -async 1 -r 25 -threads 0 {avi_file}", shell=True)
    print(f"[INFO] Extracting frames...")
    subprocess.call(f"ffmpeg -y -i {avi_file} -qscale:v 2 -threads 0 -f image2 {frames_pattern}", shell=True)
    print(f"[INFO] Extracting audio...")
    subprocess.call(f"ffmpeg -y -i {avi_file} -ac 1 -vn -acodec pcm_s16le -ar 16000 {audio_file}", shell=True)

    # FACE DETECTION
    print(f"[INFO] Running face detection...")
    faces = inference_video(cfg)
    print(f"[INFO] Detected faces in {len([f for f in faces if f])} frames")

    # SCENE DETECTION
    print(f"[INFO] Running scene detection...")
    scenes = scene_detect(cfg)
    print(f"[INFO] Found {len(scenes)} scenes")

    # FACE TRACKING
    print(f"[INFO] Running face tracking...")
    tracks = []
    for shot_idx, shot in enumerate(scenes):
        if shot[1].frame_num - shot[0].frame_num >= cfg.min_track:
            shot_tracks = track_shot(cfg, faces[shot[0].frame_num:shot[1].frame_num])
            tracks.extend(shot_tracks)
            print(f"[INFO] Shot {shot_idx}: Found {len(shot_tracks)} tracks")
    
    print(f"[INFO] Total tracks: {len(tracks)}")

    crop_faces(cfg, tracks)
    
    print(f"[INFO] Running syncnet evaluation...")
    dists = run_syncnet(cfg)

    result = {"tracks": tracks, "dists": dists}
    
    if not skip_persistent_save:
        with open(os.path.join(cfg.work_dir, reference, "results.pkl"), "wb") as f:
            pickle.dump(result, f)

    return result