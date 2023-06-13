# import lib
import io
import cv2
import streamlit as st
from inference import V5
import numpy as np
import tempfile

def load_image(infer):
    st.title("Chess Piece Object Detection via computer vision")
    file = st.file_uploader("Upload an image", type=["jpg"])
    col1, col2 = st.columns(2)

    if file:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        col1.image(image, channels="BGR", caption="Original Image", use_column_width=True)
        if st.button("Detect Piece"):
            detect = infer(image)
            col2.image(detect, channels="BGR", caption="Detected Image", use_column_width=True)

def load_video(infer):
    st.title("Real-Time detection On Video")
    file = st.file_uploader("Upload a video", type=["mp4"])
    temporary_location = False

    if file is not None:
        g = io.BytesIO(file.read())
        frame_window = st.image([])
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as out:
            out.write(g.read())
            temporary_location = out.name 
            video = cv2.VideoCapture(temporary_location)
            k=video.isOpened()
            if k==False:
                video.open(file)
            total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
            for i in range(int(total_frames)):
                ret, frame = video.read()
                if not ret:
                    break
                detect = infer(frame)
                frame_window.image(detect, channels="BGR")
                if cv2.waitKey(1) == 27:
                    break
            video.release()
            cv2.destroyAllWindows()

def webcam_stream(infer):
    st.title("Real-Time detection On Webcam")
    run = st.button('Turn On/Off Webcam')

    frame_window = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        _, frame = camera.read()
        frame_window.image(infer(frame), channels="BGR")


def main():
    st.set_page_config(page_title="Automated Chess Piece Object Detection", page_icon=":guardsman:", layout="wide")
    st.sidebar.title("Select an option")
    app_mode = st.sidebar.selectbox("Choose Source", ["Load Image", "Load Video", "Local Webcam Stream"])
    infer = V5(conf_thres=.2, iou_thres=.2)
    if app_mode == "Load Image":
        load_image(infer)
    elif app_mode == "Load Video":
        load_video(infer)
    else:
        webcam_stream(infer)

if __name__ == "__main__":
    main()
