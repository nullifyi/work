import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement = True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.33)
config.gpu_options.allow_growth = True


max_steps = 1000  # 最大迭代次数
learning_rate = 0.001   # 学习率
dropout = 0.9   # dropout时随机保留神经元的比例
data_dir = './MNIST_DATA'   # 样本数据存储的路径

# 获取数据集，并采用采用one_hot独热编码
mnist = input_data.read_data_sets(data_dir,one_hot = True)

sess = tf.InteractiveSession(config = config)

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

# 保存图像信息
with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

# 初始化权重参数
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

# 初始化偏置参数
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

# ***********请针对以下深度神经网络模型的代码段给出关键注释（42至59行）**************
# 定义一个函数，用于创建神经网络层。
# 参数包括输入张量、输入维度、输出维度、层名称和激活函数（默认为ReLU）。
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    # 使用tf.name_scope创建一个命名空间，便于在TensorBoard中组织图结构。
    with tf.name_scope(layer_name):
        # 在'weights'命名空间下创建权重变量。
        with tf.name_scope('weights'):
            # 调用weight_variable函数用于初始化权重参数。
            # [input_dim, output_dim]指定了权重矩阵的形状。
            weights = weight_variable([input_dim, output_dim])

        # 在'biases'命名空间下创建偏置变量。
        with tf.name_scope('biases'):
            # 调用bias_variable函数用于初始化偏置参数。
            # [output_dim]指定了偏置向量的形状。
            biases = bias_variable([output_dim])

        # 在'linear_compute'命名空间下进行线性计算（矩阵乘法加偏置）。
        with tf.name_scope('linear_compute'):
            # 输入张量与权重矩阵相乘，并加上偏置。
            preactivate = tf.matmul(input_tensor, weights) + biases

        # 应用非线性激活函数tf.nn.relu
        activations = act(preactivate, name='activation')

    # 返回激活后的输出。
    return activations


# 使用nn_layer函数创建第一个隐藏层。
# 输入x，输入维度784，这是图像展开后的大小，输出维度500，层名为'layer1'。
hidden1 = nn_layer(x, 784, 500, 'layer1')

# 使用tf.name_scope创建一个命名空间，用于组织dropout操作。
with tf.name_scope('dropout'):
    # 创建一个占位符，用于在训练时动态地提供dropout保留率。
    keep_prob = tf.placeholder(tf.float32)
    # 应用dropout操作到第一个隐藏层的输出上。
    # dropout是一种正则化技术，用于减少过拟合，通过随机丢弃网络中的一部分节点。
    dropped = tf.nn.dropout(hidden1, keep_prob)

# 使用nn_layer函数创建输出层（第二个隐藏层）。
# 输入是dropout后的hidden1，输入维度500，输出维度10，对应手写数字分类问题的类别数，即阿拉伯数字0~9，层名为'layer2'，激活函数设置为tf.identity（即不进行激活，直接输出）。
y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

# 创建损失函数
# ***********请介绍交叉熵损失）**************
# 交叉熵损失（Cross-Entropy Loss）是衡量两个概率分布之间差异的一种方法，常用于分类问题中。
# 在这里，tf.nn.softmax_cross_entropy_with_logits函数计算的是logits（模型的原始输出，未经过softmax归一化）和真实标签（one-hot编码）之间的交叉熵损失。
# 这个函数内部首先会对logits应用softmax函数，将其转换为概率分布，然后计算与真实标签的交叉熵。
# 交叉熵损失越小，表示模型的预测分布与真实分布越接近。
with tf.name_scope('loss'):
    # 计算交叉熵损失（每个样本都会有一个损失）
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    with tf.name_scope('total'):
        # 计算所有样本交叉熵损失的均值
        cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('loss', cross_entropy)
    
    
# 使用AdamOptimizer优化器训练模型，最小化交叉熵损失
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# 计算准确率
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        # 分别将预测和真实的标签中取出最大值的索引，弱相同则返回1(true),不同则返回0(false)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        # 求均值即为准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
