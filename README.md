# 新建python环境

(如果你想要自己配环境)
```bash
pip3 install python3.10-venv
cd eggplant_car
python -m venv eggplant_car_venv
source eggplant_car_venv/bin/activate 
```

# 安装python环境

```bash
pip3 install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple
```
然后缺什么pip3 install什么

# 下载训练权重并测试语义分割

```bash
python3 test.py
```
运行会自动下载预训练权重，模型大概200MB多

# 启动识别

```bash
python3 eggplant_detector/detector.py 
```

![](./result.png)
> 最大矩形的斜向角度: 85.68397521972656°