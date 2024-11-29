import tensorflow as tf

# ����MNIST���ݼ�
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# �ı�ά������ӦCNN���������Ҫ��
x_train4D = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') 
x_test4D = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

# ����Ԥ���� 
# *********** ���������Ԥ������ **************
# ������ֵ��0-255���ŵ�0-1֮�䣬�Ա���õ�ѵ��ģ�ͣ�ʹ��ģ�͵��ݶ��½����ȶ���
x_train, x_test = x_train4D / 255.0, x_test4D / 255.0

# ***********��������¾��������ģ�͵Ĵ���θ����ؼ�ע�ͣ�14��26�У�**************
model = tf.keras.models.Sequential([
    # ��һ�����㣬����32������ˣ�����˴�СΪ3x3��ʹ��ReLU�����
    # padding='same' ��֤���������ͼ��С��������ͬ��ͨ���ڱ�Ե����ʵ�������0
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                           input_shape=(28, 28, 1), activation='relu'),
    
    # ��һ�����ػ��㣬�ػ����ڴ�СΪ2x2�����ڼ�������ͼ�Ĵ�С�����ͼ��㸴�Ӷ�
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    # �ڶ������㣬����64������ˣ�����˴�СΪ3x3��ʹ��ReLU�����
    # padding='same' ��֤���������ͼ��С��������ͬ
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
    
    # �ڶ������ػ��㣬�ػ����ڴ�СΪ2x2�����ڽ�һ����������ͼ�Ĵ�С����ȡ���߲������
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Dropout�㣬�������25%����Ԫ����ֹ����ϣ����ģ�͵ķ�������
    tf.keras.layers.Dropout(0.25),
    
    # Flatten�㣬����ά������ͼչƽΪһά��������������ȫ���Ӳ�
    tf.keras.layers.Flatten(),
    
    # ȫ���Ӳ㣬����128����Ԫ��ʹ��ReLU�����
    tf.keras.layers.Dense(128, activation='relu'),
    
    # Dropout�㣬�������50%����Ԫ����ֹ�����
    tf.keras.layers.Dropout(0.5),
    
    # ����㣬����10����Ԫ��ʹ��softmax������������ת��Ϊ10������ĸ��ʷֲ�
    tf.keras.layers.Dense(10, activation='softmax')
])

# ���彻������ʧ����
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# �����Ż�����ѧϰ��
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# ����ģ��
model.compile(optimizer=optimizer,
              loss=loss_fn,
              metrics=['accuracy'])

# ѵ��ģ��
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# ����ģ��
model.evaluate(x_test, y_test)

