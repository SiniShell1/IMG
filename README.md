

# Inference

## directory structure
- ckpts
    - image_encoder/
    - flux_resampler.pth
    - ip_adapter.bin
    - resampler.pth
    - sdxl_resampler.pth
- LLaVA
- third-party
    - ip_adapter
    - open_clip
    - flux
    - score_models
- src
- assets

## for sdxl
1. git clone git@github.com:haotian-liu/LLaVA.git
2. cd LLaVA
3. pip install -e .
4. pip install datasets diffusers==0.32.2 ftfy protobuf==3.20
5. pip install --upgrade accelerate transformers==4.49.0 gradio

## for flux
6. pip install opencv-python optimum optimum-quanto onnxruntime
# Training

