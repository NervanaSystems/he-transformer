WEIGHT_DECAY = 1e2

class Cryptonets(model.Model):
    def __init__(self, wd=WEIGHT_DECAY, training=True):

        super(Cryptonets, self).__init__(
            model_name='cryptonets', wd=wd, training=training)

    def inference(self, images):
        conv1 = self.conv_layer(
            images,
            size=5,
            filters=20,
            stride=2,
            decay=False,
            activation=True,
            bn_before_act=True,
            name='conv1')

        pool1 = self.pool_layer(conv1, size=3, stride=1, name='pool1')

        conv2 = self.conv_layer(
            pool1,
            size=5,
            filters=50,
            stride=1,
            decay=True,
            activation=False,
            name='conv2')

        pool2 = self.pool_layer(conv2, size=3, stride=2, name='pool2')

        fc1 = self.fc_layer(pool2, neurons=100, activation=True, decay=False, bn_before_act=True, name='fc1')

        fc2 = self.fc_layer(fc1, neurons=10, activation=False, decay=False, name='fc2')

        return fc2
