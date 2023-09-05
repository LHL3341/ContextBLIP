### 实验说明
 
1. 将 $bert-base-uncased.zip$、$pos\_tag.json$ 和模型文件 $model\_base\_14M.pth$ 置于./
2. 修改预训练配置文件configs/pretrain.yaml中的train_file字段为coco.json和vg.json所在路径
3. 将coco和vg的图片放在./pretrain_data/vl_pair文件夹下，结构如图
 <img src="https://github.com/LHL3341/Adapter-BLIP/blob/main/README.assets/image-20230905205831636.png" alt="image-20230905205919235" style="zoom:50%;" />
4. 运行代码

```
unzip bert-base-uncased.zip
bash run.sh
```
