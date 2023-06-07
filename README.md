# 计算摄像学期末项目：Real-ESRGAN

Forked from https://github.com/xinntao/Real-ESRGAN. 



##### 使用说明

`inference_realesrgan.py`：用于推理。来自原作者。

按照下方代码执行，输入图片放在inputs文件夹，输出图片默认位于results文件夹。需要指定模型位置--model_path参数。例如：

```
python inference_realesrgan.py -i inputs --model_path ./net_g_10000.pth
```

`evaluation.py`：用于评估重建后的指标，如峰值信噪比（PSNR）、结构相似度、L1-loss、L2-loss。作者：xhz

`downgrade.py`：模拟图片的退化过程，用于评估前准备。作者：zyx

