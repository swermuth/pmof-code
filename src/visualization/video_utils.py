import os
import cv2
from datetime import datetime
from pathlib import Path
from IPython.display import display, clear_output, Image

from src.visualization import imgid_to_imgarray
from src.utils import logger, DATA_BASE_DIR

def save_video(frames, output_path=None, frame_limit=None, fps=10, preview=False):
    """
    Saves mp4-file for list of frames in np.array format.

    Parameters
    -------   
    frames: list[np.array]
        list of frames in np.array format
        
    output_path: str (Optional)
        Path for video file.

    frame_limit: int (Optional)
        limiting the number of frames displayed
        
    fps: int (Optional)
        Frames per second in video.

    preview: Boolean (Optional)
        Display in Jupyer Notebook.
        
    Returns
    -------
    str
        output_path of video
    """


    if frame_limit:
        frames = frames[:frame_limit]


    if not output_path:
        if not os.path.isdir("video_samples"):
            os.mkdir('video_samples')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"video_samples/{timestamp}_video.mp4"
        
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # mp4 codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # Convert back to BGR for OpenCV
        out.write(frame)
        
        if preview:
            resized = cv2.resize(frame, (320,320))
            _, frame_jpeg = cv2.imencode('.jpg', resized) # Convert the frame to JPEG format
            display(Image(data=frame_jpeg.tobytes()))
            clear_output(wait=True)

    out.release()
    
    logger.info(f"Video saved at: {output_path}")
    return output_path