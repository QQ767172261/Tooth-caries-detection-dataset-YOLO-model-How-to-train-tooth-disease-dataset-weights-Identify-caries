牙齿龋齿检测数据集，2554张，提供yolo和voc两种标注方式
1类，标注数量：
caries: 6946
image num: 2554

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cf23d09dc14b441f98bcbb5a91c98fd0.png)


### 🦷 龋齿检测数据集 (Dental Caries Detection Dataset)

| 属性 | 详细描述 |
| ------ |------ |
| **数据集名称** | 齿科龋齿目标检测数据集 |
| **图像总数** | 2,554 张 |
| **标注格式** | YOLO (.txt) / PASCAL VOC (.xml) |
| **类别数量** | 1 类 |
| **目标标签** | `caries` (龋齿) |
| **总标注框数** | 6,946 个 |

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e4e2c0bc94fa42a6a87fec4340e2b7c4.png)

### 📊 数据集详情

该数据集专注于**单类目标检测**任务，即识别口腔图像中的龋齿（蛀牙）区域。

- **数据规模**：包含 **2,554** 张经过标注的牙科图像。由于平均每张图约有 2.7 个标注框（6946/2554），表明图像中通常包含多个牙齿，且部分牙齿存在病变。
- **标注质量**：总共有 **6,946** 个标注实例，数据量充足，适合训练高精度的检测模型。
- **多格式支持**：提供 YOLO 和 VOC 两种主流格式，可直接用于 YOLO 系列（如 YOLOv5, YOLOv8）、Faster R-CNN 等不同深度学习框架的训练。

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7f18603dc3404d4d8f45fbd15c3ecf92.png)

### 💻 训练代码示例 (基于 YOLOv8)

假设你使用的是 **Ultralytics YOLOv8** 框架，且数据集已按标准结构存放。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ccce50cdb6d14d1eb28b5834cabe8fa7.png)

#### 1. 数据集目录结构示例

请确保你的数据组织如下：

```
caries_dataset/
├── images/
│   ├── train/        # 训练图片
│   ├── val/          # 验证图片
│   └── test/         # 测试图片 (可选)
└── labels/
    ├── train/        # YOLO格式的标签文件 (.txt)
    └── val/          # YOLO格式的标签文件 (.txt)
```

#### 2. 数据配置文件 (`caries.yaml`)

你需要创建一个 `caries.yaml` 文件来定义数据路径和类别：

```
# 训练和验证数据集路径
train: ./caries_dataset/images/train/
val: ./caries_dataset/images/val/

# 类别数量
nc: 1

# 类别名称
names: ['caries']
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1f722d2e8d2c4cb1b3eaa138be23a86b.png)

#### 3. Python 训练脚本

```
from ultralytics import YOLO

# 1. 加载预训练模型 (这里以 YOLOv8n 为例)
model = YOLO('yolov8n.pt')  # 你可以选择 yolov8s.pt, yolov8m.pt 等

# 2. 训练模型
results = model.train(
    data='caries.yaml',      # 指向上面创建的yaml配置文件
    epochs=100,             # 训练轮数
    imgsz=640,              # 输入图像大小
    batch=16,               # 批次大小 (根据你的GPU内存调整)
    name='caries_detection' # 实验名称
)

# 3. 验证模型
metrics = model.val()
print(f"验证集 mAP: {metrics.box.map}")
```

#### 4. 命令行训练

你也可以直接在终端运行：

```
yolo train model=yolov8n.pt data=caries.yaml epochs=100 imgsz=640
```

### 💡 建议

由于这是一个**单类别**数据集，训练时的重点应放在提高小目标（微小龋齿区域）的检测精度上。建议开启数据增强（如 `mosaic` 和 `copy_paste`），因为该数据集中平均每张图有多于 2 个目标，数据增强有助于模型更好地区分密集分布的牙齿病变。


