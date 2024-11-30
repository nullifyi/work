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
# 定义一个函数来创建神经网络层
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    # 在名为layer_name的作用域下进行操作，这样便于在可视化工具中对模型各层进行清晰展示和管理。
    with tf.name_scope(layer_name):
        # 定义该层的权重变量。
        with tf.name_scope('weights'):
            # 定义了权重的初始化方式，这里使用输入维度和输出维度
            weights = weight_variable([input_dim, output_dim])
        # 在命名空间内创建偏置项
        with tf.name_scope('biases'):
            # 定义了偏置项的初始化方式，这里使用输出维度
            biases = bias_variable([output_dim])
        # 计算线性激活前的值
        with tf.name_scope('linear_compute'):
            # 通过矩阵乘法和偏置相加来计算激活前的线性值
            preactivate = tf.matmul(input_tensor, weights) + biases
        # 应用激活函数
        activations = act(preactivate, name='activation')   # 应用激活函数，这里默认使用ReLU
    # 返回激活后的值
    return activations
        
hidden1 = nn_layer(x, 784, 500, 'layer1')

with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    dropped = tf.nn.dropout(hidden1, keep_prob)

y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

# 创建损失函数
# ***********请介绍交叉熵损失）**************
# 交叉熵损失函数衡量模型预测概率与实际发生事件概率之间的差异。
# 对于二分类问题，如果真实标签是y（0或1），模型预测的概率是p，则交叉熵损失计算为：
# L(y, p) = -[y * log(p) + (1 - y) * log(1 - p)]
# 这个公式意味着，如果模型的预测概率p与真实标签y一致，那么损失会最小（趋向于0）；
# 如果预测概率与真实标签不一致，损失会增大。

# 多分类问题的交叉熵损失
# 在多分类问题中，交叉熵损失计算所有类别的预测概率与真实标签之间的差异。
# 如果有C个类别，真实标签y是一个one-hot编码的向量，模型预测的概率分布是p，则交叉熵损失计算为：
# L(y, p) = -Σ(y_o,c * log(p_o,c))，其中y_o,c是第o个样本是否属于类别c的指示变量，
# p_o,c是模型预测第o个样本属于类别c的概率。

# 交叉熵损失的特点
# 1. 概率解释：交叉熵损失直接基于概率，提供了一个衡量预测概率分布与实际分布差异的直观方法。
# 2. 数值稳定性：在实际计算中，为了避免对数函数中的数值问题（比如对0取对数），
#   通常会对预测概率进行限制，确保它们不会精确地等于0或1。
# 3. 优化目标：在模型训练过程中，最小化交叉熵损失意味着调整模型参数，
#   使得预测的概率分布尽可能地接近于真实的概率分布。
# 4. 适用性：交叉熵损失适用于各种分类问题，无论是二分类还是多分类，
#   都可以使用交叉熵损失来衡量模型性能。
# 5. 梯度下降：由于交叉熵损失函数是可微分的，因此可以应用梯度下降等优化算法来找到最小化损失的模型参数。
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



def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        # 在名为layer_name的作用域下进行以下操作，这样便于在可视化工具（如TensorBoard）中对模型各层进行清晰展示和管理。

        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            # 在名为'weights'的子作用域内定义该层的权重变量。weight_variable函数（假设在别处定义）应该是用于创建符合指定形状的权重矩阵，
            # 这里形状为[input_dim, output_dim]，即根据输入维度和输出维度来确定权重矩阵的大小，它将用于后续的线性计算，对输入进行变换。

        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            # 在名为'biases'的子作用域内定义该层的偏置变量。bias_variable函数（同样假设在别处定义）用于创建长度为output_dim的偏置向量，
            # 偏置向量会在后续的线性计算中与经过权重变换后的输入相加，起到调整输出的作用，类似于线性方程中的截距项。

        with tf.name_scope('linear_compute'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            # 在名为'linear_compute'的子作用域内进行线性计算。首先使用tf.matmul函数将输入张量input_tensor与前面定义的权重weights进行矩阵乘法运算，
            # 这一步实现了对输入数据的线性变换，然后再加上偏置biases，得到的preactivate就是在应用激活函数之前的线性组合结果。

        activations = act(preactivate, name='activation')
            # 根据传入的激活函数act（默认为tf.nn.relu）对线性组合结果preactivate进行非线性变换。激活函数的作用是引入非线性因素，
            # 使得神经网络能够学习和表示更复杂的函数关系。这里将经过激活函数处理后的结果命名为'activation'，并返回作为该层的输出。
    return activations

hidden1 = nn_layer(x, 784, 500, 'layer1')
# 调用nn_layer函数创建第一层神经网络，输入张量为x（假设在前面已定义，应该是网络的原始输入数据，维度为784，可能是经过某种处理后的结果，比如展平后的图像数据等），
# 输入维度为784，输出维度为500，层名为'layer1'，默认使用ReLU激活函数。这一层将对输入数据进行线性变换和非线性激活处理，得到的输出赋值给hidden1。

with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    # 在名为'dropout'的作用域内，定义一个占位符keep_prob，类型为tf.float32，它将用于表示在Dropout操作中保留神经元的概率。
    # 在训练过程中，会给这个占位符传入具体的值，以控制每次迭代时神经元被保留的比例。

    dropped = tf.nn.dropout(hidden1, keep_prob)
    # 使用tf.nn.dropout函数对第一层的输出hidden1进行Dropout操作。Dropout操作会根据传入的保留神经元概率keep_prob，
    # 随机地将一部分神经元的输出设置为0，从而防止网络过拟合。得到的结果赋值给dropped，作为经过Dropout处理后的输出，将继续传递给下一层。

y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)
# 再次调用nn_layer函数创建第二层神经网络，输入为经过Dropout处理后的输出dropped，输入维度为500（与上一层的输出维度一致），
# 输出维度为10（可能对应要分类的类别数等），层名为'layer2'，这里激活函数使用tf.identity，即不进行激活函数处理（直接输出输入的值），
# 这一层将对输入数据进行进一步的线性变换和可能的激活处理（取决于具体需求，这里是直接输出），最终得到的输出赋值给y，y可能就是网络的最终输出，
# 比如在分类任务中可能表示各个类别的预测值等。



# 42. 定义一个函数来创建神经网络层
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        # 43. 在命名空间中创建权重，用于定义层的参数
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
        # 44. 在命名空间中创建偏置项
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
        # 45. 计算线性激活前的结果（权重矩阵乘以输入加上偏置）
        with tf.name_scope('linear_compute'):
            preactivate = tf.matmul(input_tensor, weights) + biases
        # 46. 应用激活函数到线性结果上
        activations = act(preactivate, name='activation')
    # 47. 返回激活后的值
    return activations

# 48. 使用nn_layer函数创建第一个隐藏层，输入维度为784，输出维度为500，层名为'layer1'
hidden1 = nn_layer(x, 784, 500, 'layer1')

# 49. 定义dropout的命名空间
with tf.name_scope('dropout'):
    # 50. 创建一个占位符，用于在训练时控制dropout的比例
    keep_prob = tf.placeholder(tf.float32)
    # 51. 应用dropout，根据keep_prob的值随机丢弃一些神经元的激活值
    dropped = tf.nn.dropout(hidden1, keep_prob)

# 52. 使用nn_layer函数创建第二个隐藏层，输入维度为500，输出维度为10，层名为'layer2'
# 53. 指定激活函数为tf.identity，即不使用激活函数，输出线性值
y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)