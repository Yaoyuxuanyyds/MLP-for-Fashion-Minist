# MLP-for-Fashion-Minist

​	

​	**本项目构建了一个基于 Python 模块化设计的多层感知机（MLP）神经网络。模型使用  Fashion-Minist 数据集进行超参数选择和训练，同时也可以尝试使用其它数据集，只需要根据数据集情况修改文件加载与预处理部分。**

---

## 1. 项目目录

- 本项目中主体为***Fashion_minist_model.ipynb***文件，模型训练与测试将在该notebook中进行。
- *module.py* 中包含了神经网络中所用到的各种**layer 对象** (Fully Connected Layer, Activation Layers, Softmax Layers 等)， 在模型的构建中被使用到。
- log 目录下存放了训练过程中根据验证集 accuracy 所保存的最优权重 (*best_model_params.npy*)，以及参数搜索得到的结果 (**hyperparameter_search_results.csv*)。
- data 目录和 utils 目录直接来自 Fashion_minist 数据集项目，包含了数据集数据和加载数据集的一些工具。
- static 目录下存放了训练过程中得到的一些可视化结果。

---

## 2. 训练说明

- 首先需要运行第一部分，加载并预处理数据集。
- **在正式训练之前可以根据自己的需要进行参数搜索**，设置合适的参数搜索范围，训练 epoch 。本项目选择了 learning rate, size of layers, L2 regularization strength 作为进行搜索的超参数，你也可以选择更多的超参数进行搜索，只需要设定搜索的范围，并为搜索代码添加相应的循环嵌套。
- 搜索结果将输出为根据模型表现排序后的列表，你可以选择合适的最优超参组合进行更多 epoch 的正式训练，获得最理想的分类模型，并保存其权重（已自动完成）用于后续模型测试。
- 模型采用 **mini-batch** 梯度优化，你也可以设定为 **batch size = 1**, 即使用 **SGD 随机梯度优化**。
- **模型支持对各种超参数的选择**，具体如下：
  - `layers`: 表示每层神经元的数量，从输入层到输出层。
  - `lr` : 学习率，决定了参数在梯度下降过程中更新的步长。
  - `activation`: 指定使用的激活函数类型，如 ReLU、Sigmoid 或 Tanh。
  - `max_epoch`: 训练的最大轮次。
  - `batch_size`: 每个mini-batch的样本数，用于小批量梯度下降。
  - `L2_lambda`: L2 正则化的强度，用于防止过拟合。
  - `decay_factor` & `decay_steps`: 学习率衰减相关参数，定期减小学习率以提高模型稳定性和性能。
  - `print_log`: 是否打印训练日志。
  - `print_iter`: 每多少批次打印一次日志信息。

- 模型另外具有**自动根据在验证集上的表现来来保存最优权重**的功能，你可以自己设定保存目录。

  

---

## 3. 测试说明

- 测试分为三个部分：**测试集 accuracy，image classification test，visualization**
- 模型测试首先创建与训练模型相同的 model_test 模型对象，并将所保存的权重加载到测试模型中。
  - 将测试集作为输入进行前向传播，计算 loss 和 accuracy。
  - 随机从测试集中选择图片进行分类，并可视化图片及预测结果。
  - 将每个全连接层的最优权重绘制成热力图，并观察其中的模式。
  - 以不同图片作为输入，将每个激活层得到的结果 reshape 为二维形式并绘制其热力图，观察不同输入下每一层激活结果的不同状态。

- 你也可以根据自己的关注兴趣为模型添加相应的测试。

---

## 4. 模型权重下载

#### 如果你选用项目第三部分（Training model with best hyperparametres) 的超参数选择，一个表现较为理想的模型权重可以在这里直接下载获取，并使用模型的 load_model 方法加载到你的模型对象中：https://drive.google.com/drive/folders/1ZwDhIhAX-pf4uBSpxF8AYU-FohFao6Jy?usp=sharing

（在 log 目录中也可以直接获取）

---

​				 				

​								   本项目为 Fudan University Computer Vision 课程作业，作者： *律己zZZ*

​	