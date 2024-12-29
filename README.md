# Project `!BoomVim`

## Getting Started

To set up and run the project, install the required dependencies using the command below:

```bash
pip install ultralytics torch torchvision scikit-learn numpy opencv-python opencv-python-headless requests PyGObject flask flask-cors pillow
```

### Important Notes:
1. **Pillow Compatibility**:  
   Due to a known compatibility issue between `Pillow` and `PyTorch`, ensure you install the specific `Pillow` version `9.5.0`:
   ```bash
   pip install pillow==9.5.0
   ```

2. **PyTorch Version**:  
   While a CPU version of PyTorch is available, we highly recommend using the latest **CUDA-enabled** PyTorch package for optimal performance.

**Note: If pip install doesn't work, you may try installing some pip packages directly from their GitHub repository.**

---

## Full List of Dependencies for the AI Server

Below is the full list of dependencies used in the AI server for the `!BoomVim` project:

```text
absl-py                 2.1.0
addict                  2.4.0
antlr4-python3-runtime  4.9.3
appdirs                 1.4.4
astunparse              1.6.3
black                   21.4b2
blinker                 1.9.0
boto3                   1.35.86
botocore                1.35.86
certifi                 2024.12.14
charset-normalizer      3.4.0
click                   8.1.8
cloudpickle             3.1.0
colorama                0.4.6
contourpy               1.3.1
cycler                  0.12.1
Cython                  3.0.11
detectron2              0.6
filelock                3.16.1
filetype                1.2.0
fire                    0.7.0
Flask                   3.1.0
flatbuffers             24.3.25
fonttools               4.55.3
fsspec                  2024.12.0
future                  1.0.0
fvcore                  0.1.5.post20221221
gast                    0.6.0
gitdb                   4.0.11
GitPython               3.1.43
google-pasta            0.2.0
grpcio                  1.68.1
h5py                    3.12.1
huggingface-hub         0.24.7
hydra-core              1.3.2
idna                    3.7
iopath                  0.1.9
itsdangerous            2.2.0
Jinja2                  3.1.5
jmespath                1.0.1
joblib                  1.4.2
keras                   3.7.0
kiwisolver              1.4.7
libclang                18.1.1
lz4                     4.3.3
Markdown                3.7
markdown-it-py          3.0.0
MarkupSafe              3.0.2
matplotlib              3.10.0
mdurl                   0.1.2
ml-dtypes               0.4.1
mmcv                    2.2.0
mmdet                   3.3.0              C:\Users\minjun\Desktop\mmdetection
mmengine                0.10.5
mpmath                  1.3.0
mtcnn                   1.0.0
mypy-extensions         1.0.0
namex                   0.0.8
networkx                3.4.2
numpy                   1.26.4
omegaconf               2.3.0
opencv-python           4.10.0.84
opencv-python-headless  4.10.0.84
opt_einsum              3.4.0
optree                  0.13.1
packaging               24.2
pandas                  2.2.3
pathspec                0.12.1
Pillow                  9.5.0
pip                     24.3.1
platformdirs            4.3.6
portalocker             3.0.0
protobuf                5.29.2
psutil                  6.1.1
py-cpuinfo              9.0.0
pybboxes                0.1.6
pycocotools             2.0.8
pydot                   3.0.3
Pygments                2.18.0
pyparsing               3.2.0
python-dateutil         2.9.0.post0
python-dotenv           1.0.1
pytz                    2024.2
pywin32                 308
PyYAML                  6.0.2
regex                   2024.11.6
requests                2.32.3
requests-toolbelt       1.0.0
rich                    13.9.4
roboflow                1.1.50
s3transfer              0.10.4
sahi                    0.11.20
scikit-learn            1.6.0
scipy                   1.14.1
seaborn                 0.13.2
setuptools              75.6.0
shapely                 2.0.6
six                     1.17.0
smmap                   5.0.1
sympy                   1.13.1
tabulate                0.9.0
tensorboard             2.18.0
tensorboard-data-server 0.7.2
tensorflow              2.18.0
tensorflow-hub          0.16.1
tensorflow_intel        2.18.0
termcolor               2.5.0
terminaltables          3.1.10
tf_keras                2.18.0
thop                    0.1.1-2209072238
threadpoolctl           3.5.0
toml                    0.10.2
torch                   2.5.1
torchaudio              2.5.1
torchvision             0.20.1
tqdm                    4.67.1
typing_extensions       4.12.2
tzdata                  2024.2
ultralytics             8.3.53
ultralytics-thop        2.0.13
UNKNOWN                 0.0.0
urllib3                 2.3.0
Werkzeug                3.1.3
wheel                   0.45.1
wrapt                   1.17.0
yacs                    0.1.8
yapf                    0.43.0
yolov5                  7.0.14
```