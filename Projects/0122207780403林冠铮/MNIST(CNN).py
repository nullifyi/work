import tensorflow as tf

# 加载MNIST数据集
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 改变维度来适应CNN网络的输入要求
x_train4D = x_train.reshape(x_train.shape[0],28,28,1).astype('float32') 
x_test4D = x_test.reshape(x_test.shape[0],28,28,1).astype('float32')

# 数据预处理 
# *********** 请介绍数据预处理步骤 **************
# 数据归一化处理
# 在将图像数据输入到神经网络之前，进行归一化是一个重要的预处理步骤。
# 归一化的目的是将原始数据的值范围调整到一个特定的尺度，通常是[0, 1]或[-1, 1]。
# 对于图像数据而言，原始像素值通常位于[0, 255]的范围内，表示图像中每个像素点的亮度。
# 归一化操作可以通过将每个像素值除以255.0来实现，从而将像素值的范围转换为[0.0, 1.0]。
# 在这个特定的例子中，x_train4D和x_test4D是四维的图像数据，分别代表训练集和测试集。
# 通过执行x_train4D / 255.0和x_test4D / 255.0，我们实际上是在对每个图像中的每个像素值进行归一化处理。
# 归一化的好处包括：
# 1. 加速模型的训练过程：归一化后的数据具有更小的范围，这有助于减少神经网络的计算量，从而加速训练过程。
# 2. 提高模型的性能：归一化后的数据在数值上更加稳定，有助于模型更好地学习和泛化。
# 3. 促进梯度下降算法的收敛：归一化有助于防止梯度消失或爆炸问题，从而有助于梯度下降算法更快地收敛到最优解。
# 因此，在这个步骤中，我们将训练集和测试集的图像数据都进行了归一化处理，以准备后续的神经网络训练过程。
x_train, x_test = x_train4D / 255.0, x_test4D / 255.0

# ***********请针对以下卷积神经网络模型的代码段给出关键注释（14至26行）**************
# 定义卷积神经网络模型
model = tf.keras.models.Sequential([
    # 第一个卷积层
    # filters=32表示使用32个滤波器（或称为卷积核）
    # kernel_size=(3,3)表示每个滤波器的大小是3x3
    # padding='same'表示在输入数据的边缘填充0，以保持输出的空间维度与输入相同
    # input_shape=(28,28,1)指定了输入数据的形状（仅在第一层需要）
    # activation='relu'表示使用ReLU激活函数
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=(28, 28, 1), activation='relu'),

    # 第一个池化层
    # 使用2x2的池化窗口，进行最大池化操作
    # 这有助于减少数据的空间维度，从而减少参数数量和计算量
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # 第二个卷积层
    # 使用64个3x3的滤波器，其他参数与第一个卷积层相同
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),

    # 第二个池化层
    # 与第一个池化层相同，使用2x2的池化窗口进行最大池化
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Dropout层
    # 在训练过程中随机丢弃25%的神经元，以减少过拟合
    tf.keras.layers.Dropout(0.25),

    # Flatten层
    # 将多维的输入一维化，以便可以用在全连接层（Dense层）
    tf.keras.layers.Flatten(),

    # 全连接层（Dense层）
    # 128个神经元，使用ReLU激活函数
    tf.keras.layers.Dense(128, activation='relu'),

    # 另一个Dropout层
    # 在全连接层后随机丢弃50%的神经元，进一步减少过拟合
    tf.keras.layers.Dropout(0.5),

    # 输出层
    # 10个神经元，对应10个类别（0-9的数字）
    # 使用softmax激活函数，将输出转换为概率分布
    tf.keras.layers.Dense(10, activation='softmax')
])

# 定义交叉熵损失函数
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# 定义优化器和学习率
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# 编译模型
model.compile(optimizer=optimizer,
              loss=loss_fn,
              metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# 评估模型
model.evaluate(x_test, y_test)
