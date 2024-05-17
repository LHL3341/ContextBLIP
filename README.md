# ContextBLIP : Doubly Contextual Alignment for Contrastive Image Retrieval from Linguistically Complex Descriptions
## Set up

```
conda create -n contextblip python=3.9
conda activate contextblip
pip install -r requirements.txt
```

## Data Preparation

COCO, VG, IMAGECODE

annotations.zip

bert-base-uncased.zip

model_base.pth

## Experiments

### Pretrain

1. unzip bert-base-uncased.zip、annotations.zip in ./
2. Modify the train_file field in the pretraining configuration file ./configs/pretrain.yaml to the list containing the paths where coco.json and vg.json reside
3. The coco and vg images are stored in the./pretrain_data/vl_pair folder, like below:
 ![1](D:\ContextBLIP\1.png)
4. run code

```
unzip bert-base-uncased.zip
unzip annotations.zip
bash run.sh
```

### Finetune
1. download imagecode dataset

   images：[image-sets.zip · BennoKrojer/ImageCoDe at main (huggingface.co)](https://huggingface.co/datasets/BennoKrojer/ImageCoDe/blob/main/image-sets.zip)

   annotations：[imagecode/data at main · McGill-NLP/imagecode (github.com)](https://github.com/McGill-NLP/imagecode/tree/main/data)

   ```
   mkdir data
   mv image-sets.zip dataset/
   mv train_data.json dataset/
   mv valid_data.json dataset/
   mv test_data_unlabeled.json dataset/
   cd dataset
   unzip image-sets.zip
   ```

   

2. 检查图片路径为./dataset/image-sets，标注路径为./dataset，如图所示

   ![2](D:\ContextBLIP\2.png)

3. run code

   ```bash
   nohup python -u finetune.py --finetuned_checkpoint_path {pretrained model path} > finetune.log 2>&1 & #开始训练
   ```

### Zero-Shot on IMAGECODE
```bash
python zero-shot_new.py --finetuned_checkpoint_path {pretrained model path}
```

### Task-specific Analysis
```bash
python analysis/analysis_finetune.py --finetuned_checkpoint_path {finetuned model path} #评估finetune模型
```

### MMVP-VLM Benchmark

you need to replace the finetune model path in line 58.

```
python evaluate_vlm_contextblip.py
```

### Comparison with GPT4

you need to replace the API Key in line 58.

```
python sample.py #sample the subsets, random seed can be changed in the file
# datapath need to be modified in the file
# GPT4 API (You Need GPT4-vision API KEY)
python GPT4v.py
# ContextBLIP
python analysis/gpt4_comparison.py
```

### Ablation Study

In the ablation experiment, the image mask rate was adjusted by adjusting the command line parameters
```bash
#img_mask_rate
nohup python -u -m torch.distributed.run --nproc_per_node 4 main.py --mask_rate ${img_mask_rate} --output_dir 'output/Pretrain/'$img_mask_rate'' > pretrain.log 2>&1 &
```
