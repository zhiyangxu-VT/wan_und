
# BLIP3o-NEXT-edit

</tiny><a href="https://jiuhaichen.github.io/BLIP3o-NEXT.github.io/" style="font-weight: normal;">Project Page</a></tiny>


<p align="center">
<img src="figure/edit_arch.png" width="1342">
</p>






 **Fully Open-Source:**
  - **Image Editing Data:** [BLIP3o-NEXT-EDIT-ENSEMBLE-DATASETS](https://huggingface.co/Salesforce/BLIP3o-NEXT-EDIT-ENSEMBLE-DATASETS/tree/main)
  - **Model Weights (3B):** [BLIP3o-NEXT-edit-VAE-Condition](https://huggingface.co/BLIP3o/BLIP3o-NEXT-edit-VAE)


🔥 Welcome to discuss with us if you have any questions.
Discord: https://discord.gg/SsVYdV84bw
or Wechat
<p align="center">
<img src="figure/wechat_2.jpg" width="256">
</p>



Install package for image editing training
```Shell
conda create -n blip3o-next python=3.11 -y
conda activate blip3o-next
pip install --upgrade pip  setuptools
pip install -r requirements.txt
pip install -e .
```

Import slurm config and environment
```Shell
TODO
```

For the inference, change the model path in inference.py and

```Shell
bash inference.sh
```

## 🚀 Training BLIP3o-NEXT-EDIT

Follow the steps below to train the **BLIP3o-NEXT-EDIT** models.

---

### 1. Download the Dataset

Download the datasets from Hugging Face:

👉 https://huggingface.co/Salesforce/BLIP3o-NEXT-EDIT-ENSEMBLE-DATASETS/tree/main

Untar the images

---

### 2. Update the Image Path Mapping

Modify the image base-path mapping dictionary in:

    blip3o/data/dataset.py

Update the mapping to your local directory with the actual image base paths.

Reference line:

    https://github.com/JiuhaiChen/BLIP3o/blob/76836b2c4cef1e5e823badd6eb68e508aff31c8e/blip3o/data/dataset.py#L71

---

### 3. Update DATA_DIR in Training Scripts

Set DATA_DIR to your BLIP3o-NEXT-EDIT-ENSEMBLE-DATASETS directory:

    scripts/blip3o_next_edit_image_id.sh
    scripts/blip3o_next_edit_image_id_with_vae_as_noise_injection_and_cross_attn.sh

---

### 4. Run Training

(A) no VAE features version:

    bash scripts/blip3o_next_edit_image_id.sh

(B) Training with VAE features (cross-attn + noise space):

    bash scripts/blip3o_next_edit_image_id_with_vae_as_noise_injection_and_cross_attn.sh

---


