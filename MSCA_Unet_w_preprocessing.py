import cv2
import scipy.spatial.distance as dist
from tensorflow.keras import layers, models, Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization, Activation, Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import SimpleITK as sitk
from functools import partial
import pickle
import math


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# class ChannelAttention(layers.Layer):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#
#         self.avg= layers.GlobalAveragePooling2D()
#         self.max= layers.GlobalMaxPooling2D()
#
#         self.fc1 = layers.Dense(in_planes//ratio, kernel_initializer='he_normal', activation='relu',
#                                 kernel_regularizer=regularizers.l2(5e-4),
#                                 use_bias=True, bias_initializer='zeros')
#         self.fc2 = layers.Dense(in_planes, kernel_initializer='he_normal',
#                                 kernel_regularizer=regularizers.l2(5e-4),
#                                 use_bias=True, bias_initializer='zeros')
#
#     def call(self, inputs):
#         avg_out = self.fc2(self.fc1(self.avg(inputs)))
#         max_out = self.fc2(self.fc1(self.max(inputs)))
#         out = avg_out + max_out
#         out = tf.nn.sigmoid(out)
#         out = layers.Reshape((1, 1, out.shape[1]))(out)
#
#         return out


class U_Net():
    def __init__(self):
        # 设置图片基本参数
        self.height = 256
        self.width = 256
        self.channels = 1
        self.shape = (self.height, self.width, self.channels)

        # 优化器
        optimizer = Adam(0.0002, 0.5)

        # u_net
        self.unet = self.build_unet()  # 创建网络变量

        self.unet.compile(loss='mse',
                          optimizer=optimizer,
                          metrics=[self.metric_fun, self.jacard_coef])
        self.unet.summary()

    def build_unet(self, n_filters=16, dropout=0.1, batchnorm=True, padding='same'):

        # 定义一个多次使用的卷积块
        def channel_attention(inputs, ratio=16):
            '''ratio代表第一个全连接层下降通道数的倍数'''

            channel = inputs.shape[-1]  # 获取输入特征图的通道数

            # 分别对输出特征图进行全局最大池化和全局平均池化
            # [h,w,c]==>[None,c]
            x_max = layers.GlobalMaxPooling2D()(inputs)
            x_avg = layers.GlobalAveragePooling2D()(inputs)

            # [None,c]==>[1,1,c]
            x_max = layers.Reshape([1, 1, channel])(x_max)  # -1代表自动寻找通道维度的大小
            x_avg = layers.Reshape([1, 1, channel])(x_avg)  # 也可以用变量channel代替-1

            # 第一个全连接层通道数下降1/4, [1,1,c]==>[1,1,c//4]
            x_max = layers.Dense(channel // ratio)(x_max)
            x_avg = layers.Dense(channel // ratio)(x_avg)

            # relu激活函数
            x_max = layers.Activation('relu')(x_max)
            x_avg = layers.Activation('relu')(x_avg)

            # 第二个全连接层上升通道数, [1,1,c//4]==>[1,1,c]
            x_max = layers.Dense(channel)(x_max)
            x_avg = layers.Dense(channel)(x_avg)

            # 结果在相叠加 [1,1,c]+[1,1,c]==>[1,1,c]
            x = layers.Add()([x_max, x_avg])

            # 经过sigmoid归一化权重
            # x = tf.nn.sigmoid(x)
            x = layers.Lambda(lambda a: tf.nn.sigmoid(a))(x)
            # 输入特征图和权重向量相乘，给每个通道赋予权重
            x = layers.Multiply()([inputs, x])  # [h,w,c]*[1,1,c]==>[h,w,c]

            return x

        def conv2d_block(input_tensor, n_filters=16, kernel_size=3, batchnorm=True, padding='same'):
            # the first layer
            x = Conv2D(n_filters, kernel_size, padding=padding)(input_tensor)
            branch1 = Conv2D(n_filters, kernel_size=1, activation="relu")

            branch2 = Sequential([
                Conv2D(n_filters, kernel_size=1, activation="relu"),
                Conv2D(n_filters, kernel_size=3, padding="SAME", activation="relu")])      # output_size= input_size

            branch3 = Sequential([
                Conv2D(n_filters, kernel_size=1, activation="relu"),
                Conv2D(n_filters, kernel_size=5, padding="SAME", activation="relu")])      # output_size= input_size

            branch4 = Sequential([
                # caution: default strides==pool_size
                # layers.MaxPool2D(pool_size=3, strides=1, padding="SAME"),
                Conv2D(n_filters, kernel_size=1, activation="relu")])                  # output_size= input_size
            branch1 = branch1(input_tensor)
            branch2 = branch2(input_tensor)
            branch3 = branch3(input_tensor)
            branch4 = branch4(input_tensor)
            outputs = layers.concatenate([branch1, branch2, branch3, branch4])
            if batchnorm:
                outputs = BatchNormalization()(outputs)
            x = Activation('relu')(outputs)

            # the second layer
            x = Conv2D(n_filters, kernel_size, padding=padding)(x)
            if batchnorm:
                x = BatchNormalization()(x)

            X = Activation('relu')(x)
            return X

        #define a channel attention module


        # 构建一个输入
        img = Input(shape=self.shape)

        # contracting path
        c1 = conv2d_block(img, n_filters=n_filters * 1,
                          kernel_size=3, batchnorm=batchnorm, padding=padding)
        a1 = channel_attention(c1)
        p1 = MaxPooling2D((2, 2))(a1)
        p1 = Dropout(dropout * 0.5)(p1)

        c2 = conv2d_block(p1, n_filters=n_filters * 2,
                          kernel_size=3, batchnorm=batchnorm, padding=padding)
        a2 = channel_attention(c2)
        p2 = MaxPooling2D((2, 2))(a2)
        p2 = Dropout(dropout)(p2)

        c3 = conv2d_block(p2, n_filters=n_filters * 4,
                          kernel_size=3, batchnorm=batchnorm, padding=padding)
        a3 = channel_attention(c3)
        p3 = MaxPooling2D((2, 2))(a3)
        p3 = Dropout(dropout)(p3)

        c4 = conv2d_block(p3, n_filters=n_filters * 8,
                          kernel_size=3, batchnorm=batchnorm, padding=padding)
        a4 = channel_attention(c4)
        p4 = MaxPooling2D((2, 2))(a4)
        p4 = Dropout(dropout)(p4)

        c5 = conv2d_block(p4, n_filters=n_filters * 16,
                          kernel_size=3, batchnorm=batchnorm, padding=padding)
        a5 = channel_attention(c5)
        # extending path
        u6 = Conv2DTranspose(n_filters * 8, (3, 3),
                             strides=(2, 2), padding='same')(a5)
        u6 = concatenate([u6, a4])
        u6 = Dropout(dropout)(u6)
        c6 = conv2d_block(u6, n_filters=n_filters * 8,
                          kernel_size=3, batchnorm=batchnorm, padding=padding)
        a6 = channel_attention(c6)

        u7 = Conv2DTranspose(n_filters * 4, (3, 3),
                             strides=(2, 2), padding='same')(a6)
        u7 = concatenate([u7, a3])
        u7 = Dropout(dropout)(u7)
        c7 = conv2d_block(u7, n_filters=n_filters * 4,
                          kernel_size=3, batchnorm=batchnorm, padding=padding)
        a7 = channel_attention(c7)

        u8 = Conv2DTranspose(n_filters * 2, (3, 3),
                             strides=(2, 2), padding='same')(a7)
        u8 = concatenate([u8, a2])
        u8 = Dropout(dropout)(u8)
        c8 = conv2d_block(u8, n_filters=n_filters * 2,
                          kernel_size=3, batchnorm=batchnorm, padding=padding)
        a8 = channel_attention(c8)

        u9 = Conv2DTranspose(n_filters * 1, (3, 3),
                             strides=(2, 2), padding='same')(a8)
        u9 = concatenate([u9, a1])
        u9 = Dropout(dropout)(u9)
        c9 = conv2d_block(u9, n_filters=n_filters * 1,
                          kernel_size=3, batchnorm=batchnorm, padding=padding)
        a9 = channel_attention(c9)
        # f9 = Conv2D(n_filters=n_filters * 1, kernel_size=3, padding=padding)
        f9 = Conv2D(filters=n_filters * 1, kernel_size=3, padding=padding)(a9)

        output = Conv2D(1, (1, 1), activation='sigmoid')(f9)

        return Model(img, output)

    def metric_fun(self, y_true, y_pred):
        fz = tf.reduce_sum(
            2 * y_true * tf.cast(tf.greater(y_pred, 0.1), tf.float32)) + 1e-8
        fm = tf.reduce_sum(
            y_true + tf.cast(tf.greater(y_pred, 0.1), tf.float32)) + 1e-8
        return fz / fm

# wo tian jia de
    def Jac(self, y_true, y_pred):
        y_pred_f = K.flatten(K.round(y_pred))
        y_true_f = K.flatten(y_true)
        num = K.sum(y_true_f * y_pred_f)
        den = K.sum(y_true_f) + K.sum(y_pred_f) - num
        return num / den

    def jacard_coef(self, y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


    def dice_coef(self, y_true, y_pred):
        y_true_f = K.flatten(y_true)  # 将 y_true 拉伸为一维.
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + 1.) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + 1.)

    def dice_coef_loss(self, y_true, y_pred):
        return 1. - self.dice_coef(y_true, y_pred)

#    ##################

    def dice_value(self, y_true, y_pred):
        fz = 2 * np.sum(y_true * y_pred)+1
        fm = np.sum(y_true) + np.sum(y_pred)+1
        return fz / fm

    def get_contours(self, img):

        # 灰度化, 二值化, 连通域分析
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(
            img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return contours[0]
        # return contours[0] if len(contours) else 0

    def hausdorff_value(self, y_true, y_pred):
        img1 = cv2.imread(y_true)
        img2 = cv2.imread(y_pred)
        cnt1 = self.get_contours(img1)
        cnt2 = self.get_contours(img2)
        # 创建计算距离对象
        #hausdorff_sd = cv2.createHausdorffDistanceExtractor()
        hausdorff_sd = sitk.HausdorffDistanceImageFilter()
        # 返回轮廓之间的距离
        return hausdorff_sd.computeDistance(cnt1, cnt2)

    def jaccard_value(self, y_true, y_pred):
        img1Data = []
        img2Data = []
        for i in range(y_true.shape[0]):
            for j in range(y_true.shape[1]):
                img1Data.append(y_true[i][j])
                img2Data.append(y_pred[i][j])
        matV = np.mat([img1Data, img2Data])
        return dist.pdist(matV, 'jaccard')[0]

    def load_data(self):
        x_train = []  # 定义一个空列表，用于保存数据集
        x_label = []
        for file in glob('./2dimage/train/*'):  # 获取文件夹名称
            for filename in glob(file + '/*'):  # 获取文件夹中的文件
                img = np.array(Image.open(filename), dtype='float32') / 255
                x_train.append(img)
        for file in glob('./2dimage/label/*'):
            for filename in glob(file + '/*'):
                img = np.array(Image.open(filename), dtype='float32') / 255
                x_label.append(img)
        x_train = np.expand_dims(np.array(x_train), axis=3)  # 扩展维度，增加第4维
        # 变为网络需要的输入维度(num, 256, 256, 1)
        x_label = np.expand_dims(np.array(x_label), axis=3)
        np.random.seed(116)  # 设置相同的随机种子，确保数据匹配
        np.random.shuffle(x_train)  # 对第一维度进行乱序
        np.random.seed(116)
        np.random.shuffle(x_label)
        # 35760，按8:2进行分配
        return x_train[:28608, :, :], x_label[:28608, :, :], x_train[28608:, :, :], x_label[28608:, :, :]

    def train(self, epochs=400, batch_size=4):
        os.makedirs('./weights_finaldeal', exist_ok=True)
        # 获得数据
        x_train, x_label, y_train, y_label = self.load_data()

        # 加载已经训练的模型
        # self.unet.load_weights(r"./weights_my/best_model_inception_chatfull_35.h5")

        # 设置训练的checkpoint
        callbacks = [EarlyStopping(patience=100, verbose=2),
                     ReduceLROnPlateau(factor=0.5, patience=10,
                                       min_lr=0.00005, verbose=2),
                     ModelCheckpoint('weights_finaldeal/best_model.h5', verbose=2, save_best_only=True)]

        # 进行训练
        results = self.unet.fit(x_train, x_label, batch_size=batch_size, epochs=epochs, verbose=2,
                                callbacks=callbacks, validation_split=0.1, shuffle=True)
        with open('log_finaldeal.pkl', 'wb') as file_txt:
            pickle.dump(results.history, file_txt)
        # 绘制损失曲线
        loss = results.history['loss']
        val_loss = results.history['val_loss']
        metric = results.history['metric_fun']
        val_metric = results.history['val_metric_fun']
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        x = np.linspace(0, len(loss), len(loss))  # 创建横坐标
        plt.subplot(121), plt.plot(x, loss, x, val_loss)
        plt.title("Loss curve"), plt.legend(['loss', 'val_loss'])
        plt.xlabel("Epochs"), plt.ylabel("loss")
        plt.subplot(122), plt.plot(x, metric, x, val_metric)
        plt.title("metric curve"), plt.legend(['metric', 'val_metric'])
        plt.xlabel("Epochs"), plt.ylabel("Dice")
        plt.show()  # 会弹出显示框，关闭之后继续运行
        fig.savefig('./evaluation/pre_MSCA_curve.png', bbox_inches='tight',
                    pad_inches=0.1)  # 保存绘制曲线的图片
        plt.close()

    def test(self, batch_size=1):
        os.makedirs('./evaluation_MSCA_pre/test_result', exist_ok=True)
        os.makedirs("./evaluation_MSCA_pre/test_png", exist_ok=True)
        self.unet.load_weights(r"./weights_finaldeal/best_model.h5")
        # 获得数据
        x_train, x_label, y_train, y_label = self.load_data()
        test_num = y_train.shape[0]
        index, step = 0, 0
        self.unet.evaluate(y_train, y_label)
        n = 0.0
        dice_sum = 0.0
        hausdorff_sum = 0.0
        jaccard_sum = 0.0
        sum = 0
        hausdorff_num = 0
        while index < test_num:
            print('schedule: %d/%d' % (index, test_num))
            step += 1  # 记录训练批数
            mask = self.unet.predict(y_train[index:index + batch_size]) > 0.1
            mask_true = y_label[index, :, :, 0]
            if (np.sum(mask) > 0) == (np.sum(mask_true) > 0):
                n += 1
            if np.sum(mask) > 0 or np.sum(mask_true) > 0:
                sum += 1
                dice_sum += self.dice_value(mask_true.squeeze(),
                                            mask.squeeze())
                img_test = Image.fromarray(np.uint8(y_train[index].squeeze() * 255))
                img_mask = Image.fromarray(np.uint8(mask_true.squeeze() * 255))
                img_pred = Image.fromarray(np.uint8(mask.squeeze() * 255))
                path_test = './evaluation_MSCA_pre/test_png/img_' + \
                            str(index) + '.png'
                path_mask = './evaluation_MSCA_pre/test_png/mask_' + \
                    str(index) + '.png'
                path_pred = './evaluation_MSCA_pre/test_png/pred_' + \
                    str(index) + '.png'
                img_test.save(path_test)
                img_mask.save(path_mask)
                img_pred.save(path_pred)
                if np.sum(mask) > 0 and np.sum(mask_true) > 0:
                    hausdorff_num += 1
                    print(hausdorff_sum)
                    #hausdorff_sum += self.hausdorff_value(path_mask, path_pred)

                    jaccard_sum += self.jaccard_value(
                        mask_true.squeeze(), mask.squeeze())
                else:
                    jaccard_sum += 0
                    hausdorff_sum += 0
            mask = Image.fromarray(np.uint8(mask[0, :, :, 0] * 255))
            mask.save('./evaluation_MSCA_pre/test_result/' + str(step) + '.png')
            mask_true = Image.fromarray(np.uint8(mask_true * 255))
            mask_true.save('./evaluation_MSCA_pre/test_result/' +
                           str(step) + 'true.png')
            index += batch_size
        acc = n / test_num
        dice = dice_sum / sum
        #hausdorff = hausdorff_sum / hausdorff_num
        jaccard = jaccard_sum / hausdorff_num
        print('the accuracy of test data is: %.4f' % acc)
        print('the dice of test data is: %.4f' % dice)
        #print('the hausdorff of test data is: %.4f' % hausdorff)
        print('the jaccard of test data is: %.4f' % (1.000-jaccard))

    def test1(self, batch_size=1):
        self.unet.load_weights(r"weights/best_model.h5")
        # 获得数据
        x_train, x_label, y_train, y_label = self.load_data()
        test_num = y_train.shape[0]
        num = 5
        for epoch in range(num):
            rand_index = []
            while len(rand_index) < 3:
                np.random.seed()
                temp = np.random.randint(0, test_num, 1)
                if np.sum(x_label[temp]) > 0:  # 确保产生有肿瘤的编号
                    self.append = rand_index.append(temp)
            rand_index = np.array(rand_index).squeeze()
            fig, ax = plt.subplots(3, 3, figsize=(18, 18))
            for i, index in enumerate(rand_index):
                mask = self.unet.predict(x_train[index:index + 1]) > 0.1
                ax[i][0].imshow(x_train[index].squeeze(), cmap='gray')
                ax[i][0].set_title('network input', fontsize=20)
                # 计算dice系数
                fz = 2 * np.sum(mask.squeeze() * x_label[index].squeeze())
                fm = np.sum(mask.squeeze()) + np.sum(x_label[index].squeeze())

                dice = fz / fm

                img_mask = Image.fromarray(np.uint8(mask.squeeze() * 255))
                img_pred = Image.fromarray(
                    np.uint8(x_label[index].squeeze() * 255))
                path_mask = 'hausdorff\\mask_' + \
                    str(index) + '.png'
                path_pred = 'hausdorff\\pred_' + \
                    str(index) + '.png'
                img_mask.save(path_mask)
                img_pred.save(path_pred)
                # hausdorff = self.hausdorff_value(path_mask, path_pred)
                # jaccard = self.jaccard_value(
                #     mask.squeeze(), x_label[index].squeeze())
                ax[i][1].imshow(mask.squeeze())
                ax[i][1].set_title('dice(%.4f),' % (
                    dice, ), fontsize=20)  # 设置title
                ax[i][2].imshow(x_label[index].squeeze())
                ax[i][2].set_title('mask label', fontsize=20)
            fig.savefig('./evaluation_MSCA_pre/show%d_%d_%d.png' % (rand_index[0], rand_index[1], rand_index[2]),
                        bbox_inches='tight', pad_inches=0.1)  # 保存绘制的图片
            print('finished epoch: %d' % epoch)
            plt.close()


if __name__ == '__main__':
    unet = U_Net()
    # unet.train()    # 开始训练网络
    unet.test()     # 评价测试集并检测测试集肿瘤分割结果
    # unet.test1()  # 随机显示
