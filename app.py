import asyncio
import logging
import queue
import threading
import urllib.request
from pathlib import Path
from typing import List, NamedTuple
from PIL import Image

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

import av
import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import tensorflow as tf

from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)

# This code is based on https://github.com/streamlit/demo-self-driving/blob/230245391f2dda0cb464008195a470751c01770b/streamlit_app.py#L48  # noqa: E501
def download_file(url, download_to: Path, expected_size=None):
    # Don't download the file twice.
    # (If possible, verify the download using the file length.)
    if download_to.exists():
        if expected_size:
            if download_to.stat().st_size == expected_size:
                return
        else:
            st.info(f"{url} is already downloaded.")
            if not st.button("Download again?"):
                return

    download_to.parent.mkdir(parents=True, exist_ok=True)

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % url)
        progress_bar = st.progress(0)
        with open(download_to, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)"
                        % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    
    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def main():

    st.header("Real Time Face Attributes Classification")

    app_face_attributes_classification()

    logger.debug("=== Alive threads ===")
    for thread in threading.enumerate():
        if thread.is_alive():
            logger.debug(f"  {thread.name} ({thread.ident})")
            
def app_face_attributes_classification():
    """Object detection demo with MobileNet SSD.
    This model and code are based on
    https://github.com/robmarkcole/object-detection-app
    """
    
    MODEL_URL = "https://github.com/oracl4/FaceAttributeClassifaction/raw/master/models/my_model.h5"  # noqa: E501
    MODEL_LOCAL_PATH = HERE / "./models/my_model.h5"

    FACE_CLASSIFIER = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"    # noqa: E501
    FACE_CLASSIFIER_LOCAL_PATH = HERE / "./models/face_detector.xml"

    CLASSES = [
        "Hat User",
        "Glasses User"
    ]

    download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=9364088)
    download_file(FACE_CLASSIFIER, FACE_CLASSIFIER_LOCAL_PATH, expected_size=930127)

    class FaceAttrClassificationVideoProcessor(VideoProcessorBase):
        confidence_threshold: float
        result_queue: "queue.Queue()"

        def __init__(self) -> None:
            self._net = tf.keras.models.load_model(MODEL_LOCAL_PATH)
            self._preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
            self._face_cascade = cv2.CascadeClassifier(str(FACE_CLASSIFIER_LOCAL_PATH))
            self.result_queue = queue.Queue()

        def _annotate_image(self, image, faces_predictions):
            
            (h, w) = image.shape[:2]
            
            # Draw the face bounding box and text
            for (x, y, w, h, prediction) in faces_predictions:

                # Text to write
                if(prediction == 0):
                    text = "You're wearing hat!"
                else:
                    text = "You're wearing glasses!"

                cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

            return image
        
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")
            (img_height, img_width) = image.shape[:2]
            
            # Get face locations
            faces = ()
            faces = self._face_cascade.detectMultiScale(image,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(100,100)
            )
            
            # Iterate through all faces and detect each face attributes
            show_image = image
            if faces is not None and len(faces) != 0:
                faces_predictions = []
                for face in faces:
                    # Get the face coordinate
                    (x, y, w, h) = face
                    
                    # Get individual faces
                    pad_size = 50

                    # Left
                    if(x - pad_size < 0):
                        x1 =  0
                    else:
                        x1 =  x - pad_size
                    
                    # Top
                    if(y - pad_size < 0):
                        y1 =  0
                    else:
                        y1 =  y - pad_size
                    
                    # Right
                    if(x + pad_size > img_width):
                        x2 =  img_width
                    else:
                        x2 =  x + w + pad_size

                    # Bottom
                    if(y + pad_size > img_height):
                        y2 =  img_height
                    else:
                        y2 =  y + h + pad_size

                    face_image = image[x1:x2, y1:y2]
                    resized = cv2.resize(face_image, (224,224), interpolation = cv2.INTER_AREA)
                    img_batch = np.expand_dims(resized, axis=0)
                    
                    prediction = self._net.predict(img_batch)
                    prediction = tf.nn.sigmoid(prediction)
                    prediction = tf.where(prediction < 0.5, 0, 1)
                    prediction = np.squeeze(prediction)
                
                    face_prediction =  np.hstack((face, prediction))
                    faces_predictions.append(face_prediction)

                annotated_image = self._annotate_image(image, faces_predictions)
                # NOTE: This `recv` method is called in another thread,
                # so it must be thread-safe.
                self.result_queue.put(faces_predictions)
                
                show_image = annotated_image
            
            return av.VideoFrame.from_ndarray(show_image, format="bgr24")
    
    webrtc_ctx = webrtc_streamer(
        key="face-attributes-classification",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=FaceAttrClassificationVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    if webrtc_ctx.state.playing:
        col1, col2, col3 = st.columns([0.2, 5, 0.2])
        col2 = col2.empty()
        table_placeholder = st.empty()
        while True:
            if webrtc_ctx.video_processor:
                try:
                    result = webrtc_ctx.video_processor.result_queue.get(
                        timeout=1.0
                    )
                    result = np.array(result)
                    average = np.average(result[:,4])
                except queue.Empty:
                    result = None
                
                if(average < 0.5):
                    col2.image(
                        (Image.open(HERE / "./res/hat_ads.jpg")),
                        use_column_width=True,
                        width=640
                    )                    
                else:
                    col2.image(
                        (Image.open(HERE / "./res/glasses_ads.jpg")),
                        use_column_width=True,
                        width=640
                    )

                table_placeholder.table(result)
            else:
                break
    
    st.markdown(
        "This demo uses the code based from streamlit-webrtc-example. "
        "https://github.com/whitphx/streamlit-webrtc-example "
        "Many thanks to the project."
    )

if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()