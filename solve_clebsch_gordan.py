import numpy as np
from spherical_harmonics import wigner_d_num
from scipy.spatial.transform import Rotation
import tensorflow as tf
from tensorflow import keras

#===============================================================================
class ClebschGordanLayer(tf.keras.layers.Layer):
    def __init__(self, n1, n2, n3):
        super(ClebschGordanLayer, self).__init__()
        self.num_outputs = 1
        #TODO: add regularizert that drives towards sum 1.
        self.cg_coeffs = self.add_weight("cg_coeffs", shape=[n1, n2, n3])

    def call(self, inputs):
        D1, D2, D3 = inputs
        # x: batch index
        lhs = tf.einsum("xai,xbj,ijc->xabc", D1, D2, self.cg_coeffs)
        rhs = tf.einsum("abi,xic->xabc", self.cg_coeffs, D3)
        residual = lhs - rhs
        return tf.reduce_mean(tf.pow(residual, 2))

#===============================================================================
def clebsch_gordan_num(l1, l2, l3, verbose=0):
    """Determine Clebsch Gordan coefficients numerically."""
    # pick random angles
    N = 100

    n1 = 2*l1 + 1
    n2 = 2*l2 + 1
    n3 = 2*l3 + 1

    D1_input = keras.layers.Input(shape=[n1, n1], name="D1")
    D2_input = keras.layers.Input(shape=[n2, n2], name="D2")
    D3_input = keras.layers.Input(shape=[n3, n3], name="D3")
    inputs = [D1_input, D2_input, D3_input]

    cg_layer = ClebschGordanLayer(n1, n2, n3)
    residual = cg_layer(inputs)

    model = keras.Model(inputs=inputs, outputs=[residual])
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(optimizer, loss=lambda a,b: b)

    samples = [[],[],[]]
    for i in range(N):
        g = np.random.rand(3) * 2 * np.pi
        D1 = wigner_d_num(g, l1)
        D2 = wigner_d_num(g, l2)
        D3 = wigner_d_num(g, l3)
        samples[0].append(D1)
        samples[1].append(D2)
        samples[2].append(D3)

    dummy = np.zeros(shape=[N,1])
    model.fit(x=samples, y=dummy, epochs=200, verbose=verbose)

    return cg_layer.cg_coeffs.numpy().copy()

#===============================================================================
def print_clebsch_gordan_num(l1,l2,l3):
    cg = clebsch_gordan_num(2,2,2)
    cg = cg / np.max(np.abs(cg))
    for i1 in range(cg.shape[0]):
        for i2 in range(cg.shape[1]):
            for i3 in range(cg.shape[2]):
                if abs(cg[i1,i2,i3]) > 1e-6:
                    print("coeffs[{},{},{}] = {}".format(i1, i2, i3, cg[i1,i2,i3]))

#EOF
