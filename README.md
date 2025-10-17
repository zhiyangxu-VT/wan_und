
# BLIP3o-NEXT-edit

</tiny><a href="https://jiuhaichen.github.io/BLIP3o-NEXT.github.io/" style="font-weight: normal;">Project Page</a></tiny>


<p align="center">
<img src="figure/edit_arch.png" width="1342">
</p>






 **Fully Open-Source:**
  - **Pretraining Data:** [27 Million Detailed Captions](https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain-Long-Caption), [5 Million Short Captions](https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain-Short-Caption)
  - **Instruction Tuning Data:** [BLIP3o-60k](https://huggingface.co/datasets/BLIP3o/BLIP3o-60k), [ShareGPT-4o-Image](https://huggingface.co/datasets/FreedomIntelligence/ShareGPT-4o-Image)
  - **Model Weights (3B):** [Pretrain](https://huggingface.co/BLIP3o/BLIP3o-NEXT-Pretrain), [Instruction Tuning](https://huggingface.co/BLIP3o/BLIP3o-NEXT-SFT), [GRPO-Geneval](https://huggingface.co/BLIP3o/BLIP3o-NEXT-GRPO-Geneval), [GRPO-Text]()
  - **Training Code:** Pretrain, Instruction Tuning, GRPO


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

