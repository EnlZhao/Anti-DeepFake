# Anti-Deepfake-Watermark

- An implementation of the paper [CMUA-Watermark: A Cross-Model Universal Adversarial Watermark for Combating Deepfakes](https://arxiv.org/abs/2105.10872) (AAAI2022).

## Configuration

支持 Mac / Linux / Windows 系统, 推荐使用 GPU 加速。

### 1. Python 环境

- Python 3.8
- install the lib by pip (推荐 conda or pipenv)

```
pip3 install -r requirements.txt
```

### 2. 准备数据集

- 下载 CelebA 数据集:

```
cd stargan
bash download.sh celeba
```

- 如果下载不成功，可以将 download.sh 中 wget 部分注释掉，手动下载压缩包并放到 stargan/data 目录下，然后运行脚本解压。

```
if [ $FILE == "celeba" ]; then

    # CelebA images and attribute labels
    URL=https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=0
    ZIP_FILE=./data/celeba.zip
    mkdir -p ./data/
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./data/
    rm $ZIP_FILE
```

下载成功后，data 目录下应该有以下文件：

```
data
└── celeba
    ├── img_align_celeba
    ├── ├── ...
    │   ├── 202598.jpg
    │   └── 202599.jpg
    └── list_attr_celeba.txt
```

> 可能需要你把文件夹的名字改一下

- 在根目录创建软连接

```
# 在 linux/MacOS 系统下
ln -s stargan/data/celeba ./data
# 在 windows 系统下 (可能需要用绝对路径)
mklink /D data stargan\data\celeba
```

创建好可以看到这样的效果：

![alt text](./readme_img.png)

### 3. 加载模型的权重

- 将[网盘](https://pan.zju.edu.cn/share/de382a9a3aaa0fc253c976b060)中的权重解压缩后移动到相应的位置（需要你先创建相应的文件夹）

```
mkdir -p ./stargan/stargan_celeba_256/models
mkdir -p ./AttentionGAN/AttentionGAN_v1_multi/checkpoints/celeba_256_pretrained
mkdir -p ./HiSD

cd Anti-Deepfake
mv ./weights/stargan/* ./stargan/stargan_celeba_256/models
mv ./weights/AttentionGAN/* ./AttentionGAN/AttentionGAN_v1_multi/checkpoints/celeba_256_pretrained
mv ./weights/HiSD/* ./HiSD
```

- 进行 inference（ 初步训练好的的 pt 模型已经放到 ready_pt 目录下），直接运行：

```
# inference in CelebA datasets with 20 images (you can change the test number in evaluate.py)
python3 inference_evalute.py <test>
```

- `test` 为可选项，如果加入 `test` 参数表示用自己刚训练好的模型进行测试（需要预先自行为训练好的模型更名），否则使用预训练模型。

- 其次你可以自己添加图片测试，**注意**：需要手动对 `inference_own_img` 中的 `c_org = torch.tensor([[0., 1., 0., 0., 1.]])` 做修改，这里的 `c_org` 是指属性标签，你需要根据你的图片属性做修改。

> 属性包括：Label: ["Black_Hair", "Blond_Hair", "Male", "Straight_Hair", "Young"]， 0 为否，1 为是。

```
# inference with your own image (one image)
python3 inference_own_img.py [path/to/your/image]
```

本仓库提供了一个 demo_input.jpg 用于测试, 直接运行即可:

```
python3 inference_own_img.py ./demo_input.jpg
```



## 训练模型

使用 NNI 工具，搜索 step size, 相应配置文件是 `nni_config.yaml` 和 `search_space.json`

```
nnictl create --config ./nni_config.yaml
```

> 具体配置需要自行搜索一下，主要是 GPU 部分

得到超参后，修改 `setting.json` 中的 step size，然后运行

```
python3 train.py
```

## Acknowledge

[StarGAN](https://github.com/yunjey/stargan), [AttentionGAN](https://github.com/Ha0Tang/AttentionGAN), [HISD](https://github.com/imlixinyang/HiSD), [nni](https://github.com/microsoft/nni), [CMUA-Watermark](https://github.com/VDIGPKU/CMUA-Watermark)
