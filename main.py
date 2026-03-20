from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uuid, shutil, os, tempfile, numpy as np
from infer import run_inference, load_models


app = FastAPI(title="SyncNet Inference API", version="1.0.0")

def convert_to_serializable(obj):
    """
    Convert numpy arrays and other non-JSON-serializable types to native Python types.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    return obj


def generate_summary(tracks, dists, frame_rate=25):
    summary = {}
    
    if len(tracks) > 0:
        total_frames = sum(len(track.get("frame", [])) for track in tracks)
        summary["faces_detected"] = len(tracks)
        summary["duration_seconds"] = total_frames / frame_rate if total_frames > 0 else 0
        summary["coverage"] = "Good" if total_frames > 100 else "Limited"
    else:
        summary["faces_detected"] = 0
        summary["duration_seconds"] = 0
        summary["coverage"] = "No faces detected"
    
    if len(dists) > 0:
        dists_array = np.array(dists)
        
        def compute_sync_accuracy(dists_array):
            GOOD_DIST = 4.0
            BAD_DIST = 10.0

            scores = []

            for d in dists_array:
                if d <= GOOD_DIST:
                    score = 100
                elif d >= BAD_DIST:
                    score = 0
                else:
                    score = 100 * (1 - (d - GOOD_DIST) / (BAD_DIST - GOOD_DIST))

                scores.append(score)

            return float(np.mean(scores))
    
        mean_accuracy = compute_sync_accuracy(dists_array)    
        if mean_accuracy >= 85:
            quality = "Excellent"
        elif mean_accuracy >= 70:
            quality = "Good"
        elif mean_accuracy >= 50:
            quality = "Fair"
        else:
            quality = "Poor"
        
        summary["lipsync_quality"] = quality
        summary["sync_accuracy_percent"] = round(mean_accuracy, 1)
        
        if len(dists_array) > 1:
            threshold = np.percentile(dists_array, 75)
        else:
            threshold = float(dists_array[0]) * 0.9
            
        problem_segments = []
        
        for idx, dist in enumerate(dists_array):
            if dist > threshold:
                segment_duration = 5
                start_time = idx * segment_duration
                end_time = start_time + segment_duration
                
                problem_segments.append({
                    "segment_index": int(idx),
                    "time_range": f"{int(start_time//60)}:{int(start_time%60):02d} - {int(end_time//60)}:{int(end_time%60):02d}",
                    "severity": "High" if dist > np.percentile(dists_array, 90) else "Medium"
                })
        
        summary["problem_areas"] = problem_segments[:5] 
        summary["total_problem_segments"] = len(problem_segments)
        
        recommendations = []
        if mean_accuracy < 50:
            recommendations.append("Video sync is significantly mismatched. Consider re-encoding or checking source files.")
        elif mean_accuracy < 70:
            recommendations.append("Multiple sync issues detected in problem areas listed above.")
        
        if len(problem_segments) > 5:
            recommendations.append(f"{len(problem_segments)} segments with sync issues detected. Review the worst ones above.")
        
        if mean_accuracy >= 70 and len(problem_segments) <= 2:
            recommendations.append("Video is well-synced overall. Good quality.")
        
        summary["recommendations"] = recommendations if recommendations else ["Video quality is acceptable."]
    else:
        summary["lipsync_quality"] = "Unable to assess"
        summary["sync_accuracy_percent"] = 0
        summary["problem_areas"] = []
        summary["total_problem_segments"] = 0
        summary["recommendations"] = ["No sync data available to analyze."]
    
    return summary

@app.on_event("startup")
async def startup_event():
    load_models(device="cuda") 
    print("[INFO] Models loaded successfully")

@app.post("/inference")
async def inference(video: UploadFile = File(...)):
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
    file_ext = os.path.splitext(video.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}"
        )
    
    ref = str(uuid.uuid4())
    
    temp_dir = tempfile.mkdtemp(prefix=f"syncnet_{ref}_")
    temp_video_path = os.path.join(temp_dir, f"video{file_ext}")
    
    try:
        with open(temp_video_path, "wb") as f:
            content = await video.read()
            f.write(content)
        
        print(f"[INFO] Processing video: {ref}")
        
        result = run_inference(temp_video_path, ref)
        
        print(f"[INFO] Inference completed for: {ref}")
        
        summary = generate_summary(result["tracks"], result["dists"])
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "reference": ref,
                "summary": summary
            }
        )
    
    except Exception as e:
        print(f"[ERROR] Inference failed for {ref}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Inference processing failed: {str(e)}"
        )
    
    finally:
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"[INFO] Cleaned up temp directory: {temp_dir}")
            except Exception as cleanup_error:
                print(f"[WARN] Failed to cleanup temp directory: {cleanup_error}")

@app.get("/health")
async def health():
    return JSONResponse(
        status_code=200,
        content={"status": "healthy", "message": "SyncNet Inference API is running"}
    )