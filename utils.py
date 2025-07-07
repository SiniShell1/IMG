import random
import torch
from transformers import AutoProcessor, AutoModel
# from PATH_CONSTANTS import *
from torch import nn
from torch.nn import functional as F
import requests
import os
import sys
from PIL import Image

from third_party.open_clip.factory import create_model_and_transforms, get_tokenizer




def get_ip_feat(image, ip_image_processor, ip_clip, ip_resampler):
    ip_image_0 = ip_image_processor(image, return_tensors="pt").pixel_values
    ip_image_0 = ip_image_0.to(ip_clip.device)
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            ip_image_1 = ip_clip(ip_image_0, output_hidden_states=True)
            ip_image_features = ip_resampler(ip_image_1.hidden_states[-2])
    return ip_image_features

def get_random_instruction(jsonfile):
    return random.choice(jsonfile)

class CLIPSelector():
    
    def __init__(self, device):
        self.device = device
        self.processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.model = AutoModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").eval().to(device)

    def score(self, images, prompt, softmax=False):

        # preprocess
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)


        with torch.no_grad():
            # embed
            image_embs = self.model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

            text_embs = self.model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

            # score
            scores =  (text_embs @ image_embs.T)[0]

            if softmax:
                scores = self.model.logit_scale.exp() * scores
                # get probabilities if you have multiple images to choose from
                probs = torch.softmax(scores, dim=-1)
                return probs.cpu().tolist()
            else:
                return scores.cpu().tolist()

class PICSelector():
    
    def __init__(self, device):
        self.device = device
        self.processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to(device)

    def score(self, images, prompt, softmax=False):

        # preprocess
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)


        with torch.no_grad():
            # embed
            image_embs = self.model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

            text_embs = self.model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

            # score
            scores =  (text_embs @ image_embs.T)[0]

            if softmax:
                scores = self.model.logit_scale.exp() * scores
                # get probabilities if you have multiple images to choose from
                probs = torch.softmax(scores, dim=-1)
                return probs.cpu().tolist()
            else:
                return scores.cpu().tolist()
            
class AESMLP(nn.Module):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


class AESSelector():
    
    def __init__(self, device):
        self.device = device

        self.model = AESMLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14

        s = torch.load("third_party/score_models/sac+logos+ava1-l14-linearMSE.pth")   # load the model you trained previously or the model available in this repo
        self.model.load_state_dict(s)

        self.model.to(device)
        self.model.eval()

        clip_model_name = "openai/clip-vit-large-patch14"
        self.model2 = AutoModel.from_pretrained(clip_model_name).eval().to(device)
        self.processor = AutoProcessor.from_pretrained(clip_model_name)
        # self.model2, self.preprocess = clip.load("ViT-L/14", device=device)  #RN50x64   

    def score(self, images, prompt_not_used=None):
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            # embed
            image_embs = self.model2.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
            return self.model(image_embs).cpu().flatten().tolist()

root_path = "third_party/score_models"
HF_HUB_PREFIX = 'hf-hub:'

class HPSSelector():
    
    def __init__(self, device):
        self.device = device 
        model, preprocess_train, self.preprocess_val = create_model_and_transforms(
            'ViT-H-14',
            'laion2B-s32B-b79K',
            precision='amp',
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )

        # check if the default checkpoint exists
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        checkpoint_path = os.path.join(root_path, 'HPS_v2_compressed.pt')
        if not os.path.exists(checkpoint_path):
            print('Downloading HPS_v2_compressed.pt ...')
            url = 'https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt'
            r = requests.get(url, stream=True)
            with open(os.path.join(root_path, 'HPS_v2_compressed.pt'), 'wb') as HPSv2:
                total_length = int(r.headers.get('content-length'))
                for chunk in r.iter_content(chunk_size=1024): 
                    if chunk:
                        HPSv2.write(chunk)
                        HPSv2.flush()
            print('Download HPS_v2_compressed.pt to {} sucessfully.'.format(root_path+'/'))


        print('Loading model ...')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        self.tokenizer = get_tokenizer('ViT-H-14')
        model = model.to(device)
        model.eval()
        self.model = model
        print('Loading model successfully!')

    def score(self, img_path, prompt):
        if isinstance(img_path, list):
            result = []
            for one_img_path in img_path:
                # Load your image and prompt
                with torch.no_grad():
                    # Process the image
                    if isinstance(one_img_path, str):
                        image = self.preprocess_val(Image.open(one_img_path)).unsqueeze(0).to(device=self.device, non_blocking=True)
                    elif isinstance(one_img_path, Image.Image):
                        image = self.preprocess_val(one_img_path).unsqueeze(0).to(device=self.device, non_blocking=True)
                    else:
                        raise TypeError('The type of parameter img_path is illegal.')
                    # Process the prompt
                    text = self.tokenizer([prompt]).to(device=self.device, non_blocking=True)
                    # Calculate the HPS
                    if True: # with torch.cuda.amp.autocast():
                        outputs = self.model(image, text)
                        image_features, text_features = outputs["image_features"], outputs["text_features"]
                        logits_per_image = image_features @ text_features.T

                        hps_score = torch.diagonal(logits_per_image).cpu().numpy()
                result.append(hps_score[0])    
            return result
        elif isinstance(img_path, str):
            # Load your image and prompt
            with torch.no_grad():
                # Process the image
                image = self.preprocess_val(Image.open(img_path)).unsqueeze(0).to(device=self.device, non_blocking=True)
                # Process the prompt
                text = self.tokenizer([prompt]).to(device=self.device, non_blocking=True)
                # Calculate the HPS
                if True: # with torch.cuda.amp.autocast():
                    outputs = self.model(image, text)
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    logits_per_image = image_features @ text_features.T

                    hps_score = torch.diagonal(logits_per_image).cpu().numpy()
            return [hps_score[0]]