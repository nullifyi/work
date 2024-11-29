import tensorflow as tf

# 加载MNIST数据集
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 改变维度来适应CNN网络的输入要求
x_train4D = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') 
x_test4D = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

# 数据预处理 
# *********** 请介绍数据预处理步骤 **************
# 将像素值从0-255缩放到0-1之间，以便更好地训练模型，使得模型的梯度下降更稳定。
x_train, x_test = x_train4D / 255.0, x_test4D / 255.0

# ***********请针对以下卷积神经网络模型的代码段给出关键注释（14至26行）**************
model = tf.keras.models.Sequential([
    # 第一层卷积层，包含32个卷积核，卷积核大小为3x3，使用ReLU激活函数
    # padding='same' 保证输出的特征图大小与输入相同，通过在边缘添加适当数量的0
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                           input_shape=(28, 28, 1), activation='relu'),
    
    # 第一层最大池化层，池化窗口大小为2x2，用于减少特征图的大小，降低计算复杂度
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    # 第二层卷积层，包含64个卷积核，卷积核大小为3x3，使用ReLU激活函数
    # padding='same' 保证输出的特征图大小与输入相同
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
    
    # 第二层最大池化层，池化窗口大小为2x2，用于进一步减少特征图的大小，提取更高层的特征
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Dropout层，随机丢弃25%的神经元，防止过拟合，提高模型的泛化能力
    tf.keras.layers.Dropout(0.25),
    
    # Flatten层，将三维的特征图展平为一维向量，用于输入全连接层
    tf.keras.layers.Flatten(),
    
    # 全连接层，包含128个神经元，使用ReLU激活函数
    tf.keras.layers.Dense(128, activation='relu'),
    
    # Dropout层，随机丢弃50%的神经元，防止过拟合
    tf.keras.layers.Dropout(0.5),
    
    # 输出层，包含10个神经元，使用softmax激活函数，将结果转化为10个分类的概率分布
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

