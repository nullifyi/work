import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)
config.gpu_options.allow_growth = True

max_steps = 1000  # 最大迭代次数
learning_rate = 0.001   # 学习率
dropout = 0.9   # dropout时随机保留神经元的比例
data_dir = './MNIST_DATA'   # 样本数据存储的路径

# 获取数据集，并采用one_hot独热编码
mnist = input_data.read_data_sets(data_dir, one_hot=True)

sess = tf.InteractiveSession(config=config)

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

# 保存图像信息
with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

# 初始化权重参数
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# 初始化偏置参数
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# ***********请针对以下深度神经网络模型的代码段给出关键注释（42至59行）**************
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        # 初始化权重矩阵，维度为 [input_dim, output_dim]
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
        # 初始化偏置向量，维度为 [output_dim]
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
        # 线性计算部分，进行矩阵乘法并加上偏置
        with tf.name_scope('linear_compute'):
            preactivate = tf.matmul(input_tensor, weights) + biases
        # 激活函数部分，对线性输出应用激活函数
        activations = act(preactivate, name='activation')
    return activations

# 构建第一层隐藏层，输入维度为 784（28x28），输出维度为 500，使用 ReLU 激活函数
hidden1 = nn_layer(x, 784, 500, 'layer1')

# 应用 Dropout 技术，防止过拟合
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    dropped = tf.nn.dropout(hidden1, keep_prob)

# 构建输出层，输入维度为 500，输出维度为 10，使用 identity 作为激活函数
y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

# 创建损失函数
# ***********请介绍交叉熵损失）**************
with tf.name_scope('loss'):
    # 计算交叉熵损失（每个样本都会有一个损失）
    # 交叉熵损失用于衡量模型预测的类别概率分布与真实标签之间的差距
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
        # 分别将预测和真实的标签中取出最大值的索引，相同则返回1(True)，不同则返回0(False)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        # 求均值即为准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
S
