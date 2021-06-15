# watermark-detection-and-removal

> Unofficial implementation of 《Towards Photo-Realistic Visible Watermark Removal with Conditional Generative Adversarial Networks》
>
> Source: 浙江大学SRTP项目——水印检测与去除
>
> Contributors: Xiaodan Xu, Yining Mao, Hanjing Zhou

## 说明

本项目为《Towards Photo-Realistic Visible Watermark Removal with Conditional Generative Adversarial Networks》一文的非官方复现。

**注意：代码还不完善，生成器效果还不太好，需要继续调整超参（学习率、各项loss权重等等），代码仅供参考哈**

**本项目仅供学术研究参考，禁止用于任何商业用途！**



## 我们的其他去水印工作

我们的另一个项目“半自动水印检测与去除”：

[semi-automatic-watermark-removal](https://github.com/doriscullen/semi-automatic-watermark-removal)

注：此项目基于GitHub项目[automatic-watermark-detection](https://github.com/rohitrango/automatic-watermark-detection)修改，在此对项目作者表示感谢！

automatic-watermark-detection项目是对paper "On The Effectiveness Of Visible Watermarks"的非官方复现。我们在项目作者工作的基础上，完成了随机位置水印检测与去除、满屏水印检测与去除的工作，欢迎感兴趣者参考与批评指正。



## 水印研究数据集

水印研究数据集包含15000张带文字水印图和1000张带满屏水印图

您可以从百度网盘下载我们创建的水印研究数据集：

【链接：https://pan.baidu.com/s/14Ap00RcrfhlEsO3WBB62_Q	提取码：syyj】



## 本项目使用说明

### 1 加水印程序

您可在"add_watermark"文件夹下找到加水印程序的代码。

注意：代码中，path为原图文件夹路径，outpath为输出的裁剪后的原图文件夹路径（程序将原图中的图片裁剪为300*300的大小，并重命名为1.png, 2.png ...），savePath则为添加水印后的图片的存访路径。

#### 满屏水印

源码：add_image_watermark.py

水印图：multiple.png

使用方法：python add_image_watermark.py

#### 随机位置文字水印

源码：add_random_text_watermark.py

字体文件：Antonio-Regular.ttf

使用方法：python add_random_text_watermark.py

### 2 去水印cGAN模型训练

**请先下载VGG16模型的预训练参数“vgg16-397923af.pth”，并放在项目的根目录下**

【链接：https://pan.baidu.com/s/1N8A3BAEQ6j0K091v_5obpg 	提取码：vgg1 】

请将训练集、测试集图片放置dataset文件夹下对应的文件夹中。注意，如果您想直接使用我们的程序进行训练，请参照我们的图片命名方式，否则，您需要修改程序中的对应部分，以加载您自己的数据集。请将N改为正确的图片数。

在本项目中，用于训练的图片大小为256*256。如果您需要修改图片大小，请修改代码中的image_size、img_shape、whc。其中whc为VGG16网络relu2_2层输出的维度。

我们默认每5轮保存一次效果图和模型参数，您可在源文件中进行修改。

使用方法：python remove_watermark.py

结果保存位置：result

### 3 测试模型效果

请将N改为正确的图片数。

"model_parameters"文件夹下存放了我们用12000张图训练330轮的模型的参数，您可以尝试加载它并进行测试。

使用方法：python test.py

测试结果保存位置：test_result



## 展望

**我们的模型还有待改善，您可以思考以下方面**

- 扩大训练集（12000->3w+张；给图片加上不同的文字或图片水印）

- 学习率衰减（在训练初期学习率大一些，使得网络收敛迅速，在训练后期学习率小一些，使得网络更好的收敛到最优解）

- 调整L1 loss、GAN loss、perceptual loss的权重

- 调整其他参数……

