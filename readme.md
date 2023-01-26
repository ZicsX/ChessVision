# Visual Pollution Detection

This project is about Pothole severity classification via computer vision. We have used object detection algorithms to detect different types of Pothole and classify them. The algorithm is trained using various datasets and converted into onnx format for faster computation.

The following libraries are required to run the project:

- Torch
- Opencv
- Streamlit
- ONNX Runtime

## Installation

The project is written in Python and can be installed using the following command:

```bash
python  -m venv env
env/Scripts/activate
pip install -r .\requirements.txt
```

To run the web app use the following command:

```bash
streamlit run app.py
``
