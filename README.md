# PlaneObjectDetect
使用最容易的Two Stage策略进行飞机识别，本识别代码可以轻松移植到任何的目标检测任务中，只需要你重新训练CNN网络
# 效果
输入图像：
![Aaron Swartz](https://raw.githubusercontent.com/jiangruoqiao/PlaneObjectDetect/master/ObjectDectectDemo/image/input.JPG)
结果图像：
![Aaron Swartz](https://raw.githubusercontent.com/jiangruoqiao/PlaneObjectDetect/master/ObjectDectectDemo/image/result.png)
# 说明
使用Seletive Search进行图像分割，对于遥感图像目标来说由于具有旋转，所以使用TI-Pooling网络进行目标分类
# 使用
直接运行jiayoujidemo.py即可运行，使用前请确定图片目录正确
# 软件版本
Python 2.7  
Tensorflow-gpu 1.4  
Opencv  
Pillow  
Skimage
# 扩展
如果你需要将此扩展到其他图像目标的识别，请重新训练CNN模型，并进行更换，对于遥感图像目标检测推荐识别旋转不变性网络进行训练，如有需要请参考本人另外一个Github项目，复现西北工业大学自动化学院老师的RICNN网络
