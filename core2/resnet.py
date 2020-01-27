import tensorflow as tf
import scipy.io


resnet_layers = ['conv1_1', 'relu1_1', 'pool1',
                 'conv_block1_1', 'id_block1_1', 'id_block1_2',
                 'conv_block2_1', 'id_block2_1', 'id_block2_2', 'id_block2_3',
                 'conv_bolck3_1', 'id_block3_1', 'id_block3_2', 'id_block3_3', 'id_block_3_4',
                 'conv_block4_1', 'id_block4_1', 'id_block4_2',
                 'avg_pool1', 'flatten1', 'dense1']


vgg_layers = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
              'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
              'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
              'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
              'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4']

class ResNet(object):
    def __init__(self, vgg_path):
        self.vgg_path = vgg_path

    def build_inputs(self):
        self.images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'images')

    def build_params(self):
        model = scipy.io.loadmat(self.vgg_path)
        layers = model['layers'][0]
        self.params = {}
        with tf.variable_scope('encoder'):
            for i, layer in enumerate(layers):
                layer_name = layer[0][0][0][0]
                layer_type = layer[0][0][1][0]
                if layer_type == 'conv':
                    w = layer[0][0][2][0][0].transpose(1, 0, 2, 3)
                    b = layer[0][0][2][0][1].reshape(-1)
                    self.params[layer_name] = {}
                    self.params[layer_name]['w'] = tf.get_variable(layer_name+'/w', initializer=tf.constant(w))
                    self.params[layer_name]['b'] = tf.get_variable(layer_name+'/b',initializer=tf.constant(b))

    def _dense(self, x):
        return tf.contrib.layers.fully_connected(x, [-1, 196, 512], activation='softmax') 

    def _avg_pool(self, x):
        return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID') 

    def _flatten(self, x):
        return tf.contrib.layers.flatten(x)

    def _conv(self, x, w, b):
        return tf.nn.bias_add(tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME'), b)

    def _relu(self, x):
        return tf.nn.relu(x)

    def _max_pool(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    def _identity_block(self, x, filters):
        
        pass

    def build_model(self):
        for i, layer in enumerate(vgg_layers):
            filters = [64, 64, 256]
            #layer_type = layer[:4]
            if 'conv' in layer_type:
                if layer == 'conv1_1':
                    h = self.images
                h = self._conv(h, self.params[layer]['w'], self.params[layer]['b'])
            elif 'relu' in layer_type:
                h = self._relu(h)
            elif 'max_pool' in layer_type:
                h = self._pool(h)
            elif 'avg_pool' in layer_type:
                h = self._avg_pool(h)
            elif 'flatten' in layer_type:
                h = self._flatten(h)
            elif 'dense' in layer_type:
                h = self._dense(h)
            if layer == 'conv5_3':
                self.features = tf.reshape(h, [-1, 196, 512])
            

    def build(self):
        self.build_inputs()
        self.build_params()
        self.build_model()
