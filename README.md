### 实验说明

1. unzip  $bert-base-uncased.zip$
2. 将$pos\_tag.json$和模型文件$model\_base\_14M.pth$放在此目录下
3.  修改预训练配置文件configs/pretrain.yaml中的train_file字段为coco.json和vg.json所在路径
4. 将coco和vg的图片放在./pretrain_data/vl_pair文件夹下，结构如图
5. <img src="F:\cv_nlp\Adapter-BLIP\README.assets\image-20230905205919235.png" alt="image-20230905205919235" style="zoom:50%;" />
6. 运行代码

```
bash run.sh
```
