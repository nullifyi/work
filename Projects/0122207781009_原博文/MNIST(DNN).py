import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)
config.gpu_options.allow_growth = True

max_steps = 1000  # ����������
learning_rate = 0.001   # ѧϰ��
dropout = 0.9   # dropoutʱ���������Ԫ�ı���
data_dir = './MNIST_DATA'   # �������ݴ洢��·��

# ��ȡ���ݼ���������one_hot���ȱ���
mnist = input_data.read_data_sets(data_dir, one_hot=True)

sess = tf.InteractiveSession(config=config)

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

# ����ͼ����Ϣ
with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

# ��ʼ��Ȩ�ز���
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# ��ʼ��ƫ�ò���
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# ***********������������������ģ�͵Ĵ���θ����ؼ�ע�ͣ�42��59�У�**************
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        # ��ʼ��Ȩ�ؾ���ά��Ϊ [input_dim, output_dim]
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
        # ��ʼ��ƫ��������ά��Ϊ [output_dim]
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
        # ���Լ��㲿�֣����о���˷�������ƫ��
        with tf.name_scope('linear_compute'):
            preactivate = tf.matmul(input_tensor, weights) + biases
        # ��������֣����������Ӧ�ü����
        activations = act(preactivate, name='activation')
    return activations

# ������һ�����ز㣬����ά��Ϊ 784��28x28�������ά��Ϊ 500��ʹ�� ReLU �����
hidden1 = nn_layer(x, 784, 500, 'layer1')

# Ӧ�� Dropout ��������ֹ�����
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    dropped = tf.nn.dropout(hidden1, keep_prob)

# ��������㣬����ά��Ϊ 500�����ά��Ϊ 10��ʹ�� identity ��Ϊ�����
y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

# ������ʧ����
# ***********����ܽ�������ʧ��**************
with tf.name_scope('loss'):
    # ���㽻������ʧ��ÿ������������һ����ʧ��
    # ��������ʧ���ں���ģ��Ԥ��������ʷֲ�����ʵ��ǩ֮��Ĳ��
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    with tf.name_scope('total'):
        # ��������������������ʧ�ľ�ֵ
        cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('loss', cross_entropy)

# ʹ��AdamOptimizer�Ż���ѵ��ģ�ͣ���С����������ʧ
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# ����׼ȷ��
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        # �ֱ�Ԥ�����ʵ�ı�ǩ��ȡ�����ֵ����������ͬ�򷵻�1(True)����ͬ�򷵻�0(False)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        # ���ֵ��Ϊ׼ȷ��
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
S
