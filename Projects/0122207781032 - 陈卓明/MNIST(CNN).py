import tensorflow as tf

# 加载MNIST数据集
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 改变维度来适应CNN网络的输入要求
x_train4D = x_train.reshape(x_train.shape[0],28,28,1).astype('float32') 
x_test4D = x_test.reshape(x_test.shape[0],28,28,1).astype('float32')

# 数据预处理 
# *********** 请介绍数据预处理步骤 **************
# 此处进行了归一化的预处理：
# 代码 x_train4D / 255.0 和 x_test4D / 255.0 将训练集和测试集中的图像数据从原始的[0, 255]像素值范围归一化到[0, 1]的浮点数范围。
# 这里的归一化是通过将每个像素值除以255（即最大像素值）来实现的，因为图像数据是8位的，像素值范围从0到255。
x_train, x_test = x_train4D / 255.0, x_test4D / 255.0

# ***********请针对以下卷积神经网络模型的代码段给出关键注释（14至26行）**************
# 创建一个顺序模型，用于线性堆叠网络层
model = tf.keras.models.Sequential([
    # 第一个卷积层，使用32个3x3的卷积核，激活函数为ReLU
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same',
                 input_shape=(28,28,1),  activation='relu'),
    # 第一个最大池化层，池化窗口大小为2x2
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    # 第二个卷积层，使用64个3x3的卷积核，激活函数为ReLU
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', 
    			 activation='relu'),
    # 第二个最大池化层，池化窗口大小为2x2
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    # Dropout层，用于减少过拟合，随机丢弃25%的神经元
    tf.keras.layers.Dropout(0.25),
    # Flatten层，将多维输入展平为一维
    tf.keras.layers.Flatten(),
    # 全连接层，有128个神经元，激活函数为ReLU
    tf.keras.layers.Dense(128, activation='relu'),
    # 另一个Dropout层，用于减少过拟合，随机丢弃50%的神经元
    tf.keras.layers.Dropout(0.5),
    # 输出层，有10个神经元对应10个类别，使用softmax激活函数输出概率分布
    tf.keras.layers.Dense(10,activation='softmax')
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
