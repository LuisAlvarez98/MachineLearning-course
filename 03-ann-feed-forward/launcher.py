"""
Modified by:
- Jesús Omar Cuenca Espino    A01378844
- Luis Felipe Alvarez Sanchez A01194173
- Juan José González Andrews  A01194101
- Rodrigo Montemayor Faudoa   A00821976

Date: 25/09/2021
"""

import matplotlib.pyplot as plt
from ann import NeuralNetwork
import numpy as np
from utils import get_one_hot, plot_decision_boundary, read_dataset

def run_for_dataset(dataset, hidden_layers_conf, theta_config, examples):
    X, y = read_dataset(dataset)
    unique_classes = len(np.unique(y))
    y = get_one_hot(y, unique_classes)
    num_features = X.shape[0]

    ann = NeuralNetwork(input_neurons=num_features,
                        hidden_layers_conf=hidden_layers_conf,
                        output_neurons=unique_classes,
                        learning_rate=0.01,
                        regularization_rate=0.01,
                        epochs=0)
    ann.theta = theta_config

    for example in examples:
        r = ann.predict(example)
        print("{} predicted as {}".format(example.T, r.T))

    plot_decision_boundary(
        X.T, y.T, ann, title="ANN dataset={}".format(dataset))

def theta_config_for_xnor():
    #
    # B1   B2   --
    # 
    # X1   U1   U1
    # 
    # X2   U2   

    theta_config = [
        # Layer 2: 2 x 3: Arriving to Layer 2 with 2 neurons. Layer 1 (input) has 3 units (2 features + bias)
        np.array([
            [-30,  20,  20],
            [10,  -20, -20]
        ]),
        # Layer 3: 1 x 3: Arriving to Layer 3 (output) with 1 neuron. Previous layer (Layer 2) has 3 units
        np.array([
            [-10,  20, 20]
        ])
    ]
    return theta_config

def theta_config_for_blobs():
    #
    # B1   B2   B3   --
    # 
    # X1   U1   U1   U1
    # 
    # X2   U2   U2   U2
    #     
    #           U3   U3
    #           U4
    #           U5

    theta_config = [
        # Layer 2: 2 x 3: Arriving to Layer 2 with 2 neurons. Layer 1  (input) has 3 units (2 features + bias)
        np.array([
            [0.73631465,  1.34022596,  1.16982023],
            [0.42420571,  0.50366857, -0.58967897]
        ]),
        # Layer 3: 5 x 3: Arriving to Layer 3 with 5 neurons. previous layer (Layer 2) has 3 units
        np.array([
            [0.15703111,  0.70441792,  2.14853411],
            [0.64368206, -2.79339583,  0.86212386],
            [0.31886793,  1.87067951, -2.11767332],
            [0.89068826, -2.37237846, -2.95083085],
            [0.8113844,  1.24975553, -0.48260608]
        ]),
        # Layer 3: 3 x 6: Arriving to Layer 4 (output) with 3 neurons as there are 3 classes. previous layer (Layer 2) has 6 units (5+bias)
        np.array([
            [0.4840603,  1.59424142,  1.4280795, -3.07636732, -2.50115274, -1.32687562],
            [0.11750783, -0.63018121, -3.72556311, 2.15029477, -1.5928429, 0.80499961],
            [0.51071352, -3.03777759,  1.03425309, -0.13150582,  3.71668251, -1.14350344]
        ])
    ]
    return theta_config

def theta_config_for_moons():
    theta_config = [
        np.array([
            [ 0.73631465,  7.64131006,  1.25697442],
            [ 0.42420571, -3.38692533, -0.43870295],
            [ 0.15703111, -0.65094581,  0.69250166],
            [ 0.64368206, -0.71174519,  0.64052388]
        ]), 
        np.array([
            [ 0.31886793, -0.76094424, -1.13874857,  0.97845676,  0.94572946],
            [ 0.08510503,  4.56543543,  9.60621867, -6.64583929, -6.34967783],
            [ 0.7421301 , -2.50530467, -3.67418881,  2.00340246,  2.13696247],
            [ 0.11750783, -3.58808154, -7.20056362,  5.00245734,  4.55840469]
        ]), 
        np.array([
            [  0.97074786,  -1.12782273,   8.52792101,  -3.72491613, -6.57914164],
            [  0.75592653,  -0.14558487,   1.99151057,   0.03577254, -0.98759179],
            [  0.50637582,   0.11256893,   0.15738719,   0.74892826, 0.93385014],
            [  0.24107067,   0.81991764, -12.17298596,   3.80037102, 8.43308512]
        ]), 
        np.array([
            [  0.71597823, -10.1709838 ,  -2.90080894,  -0.44997138, 13.24597421],
            [  0.90447715,   9.59041081,   2.12363741,  -0.11071204, -13.78129121]
        ]
    )]
    return theta_config

def theta_config_for_circles():
    theta_config = [
        np.array([
            [ 0.73631465,  1.66271189,  3.73888663],
            [ 0.42420571,  2.80790311, -0.78398977],
            [ 0.15703111,  2.32677535, -0.60877878],
            [ 0.64368206, -2.06147134,  1.25909878],
            [ 0.31886793,  0.75798529,  1.82083411],
            [ 0.89068826, -0.22166331, -3.01451118],
            [ 0.8113844 , -0.17395305, -3.05635383],
            [ 0.4840603 , -7.29228742,  2.2669435 ],
            [ 0.0351575 , -2.17763994,  0.70183991]
        ]), 
        np.array([
            [ 0.11750783,   6.53265563,   9.93600145,   8.26150048, -5.53626372,   6.78490742,   7.3247051 ,   6.2877528,    -6.86599102, -10.33651059],
            [ 0.75592653,  -3.69286118,   5.75442724,   3.5918352,  -5.09586668,  -8.42379468,  14.46499735,  14.50972747,    7.80153658,  -0.03133771],
            [ 0.24107067,   2.46319183,  -5.47519188,  -5.4996347,  12.57343873,  -0.71199851,   2.14827345,   3.34095462,    9.32095269,   9.50779828],
            [ 0.90447715,   3.14074367,  -4.44064916,  -6.27948586, -0.01973424,   0.56427554,   2.88026533,   4.79900716,    7.54533147,   7.32107188],
            [ 0.38096354,  -8.51482144,  -7.62175878,  -7.70668176, -0.37584735,  -2.64358679,  -6.97562663,  -5.77384967,    8.27234021,   8.16671286],
            [ 0.92464576,  -5.97242926,   7.03832073,   8.73803456, -4.32276143,  -1.47363079,  -4.32133339,  -5.7401399,    -9.82816123,  -8.56524542],
            [ 0.13388894,  16.30595158,   4.52793725,   0.41177179,  1.48192172,   14.63376679,  -4.95376167,  -5.02455046,   3.11497131,  -1.33563049],
            [ 0.09531394,  -2.45037776,   3.2552466 ,   3.41424685, -12.724059,     3.58893613,  -0.94740307,  -1.53294045, -15.83203206,  -9.959501],
            [ 0.10917502,   7.29917081,   7.84664029,   6.87444653, -3.81561566,   4.29371905,   7.07526074,   6.23283164,   -4.64141538,  -7.61830878]
        ]), 
        np.array([
            [ 0.2779539 ,   3.89272708,  -7.71956391,  -4.42748898, -0.93814368,  -0.19275984,   8.72605982,   0.28922457, 14.92607427,   4.61844526],
            [ 0.06098851,  -0.47734161,  -2.3833359 ,  -7.75900279, -4.69205831,  -4.15910061,   8.13547373,  -1.16130318,  8.33914626,  -0.1411186 ],
            [ 0.84361359,   2.82259702,  -6.23083413,   6.26640732,  6.11390726,  12.15952945,   0.86597867, -11.35596571, -3.63381748,  -5.25477833],
            [ 0.68808941,  -2.25131949,   3.71108266,  -0.75646383,  1.85199838,  -1.05804094,  -2.11865454,  -0.99339901,  0.51887789,   0.69037741],
            [ 0.85098844,  -0.94614012,  -1.25688187,   1.42920446,  0.68499196,   5.04344181,   0.79648574,  -2.34281616, -0.76521018,  -4.96558061],
            [ 0.21206596,  -1.09504333,  -2.75450945,  -7.81833035, -5.31996454,  -4.03901514,   9.92163506,  -2.07557232, 10.4233922 ,  -0.35098337],
            [ 0.75972725,   4.7916578 ,  -8.45944941,   8.31092544, -8.23711507,  -2.7619424 ,   2.65627141,  -6.86571367,-13.41585197,   5.22329196],
            [ 0.93235307,   2.4697747 , -11.26636661,   5.82405692,  4.16078444,  11.80558072,   2.14233975,  -9.44238599, -4.1008085 ,   1.47643917],
            [ 0.45910788,  15.29901917,  -4.89778135,  -6.87240628, -6.44118965,  -2.8983133 ,   0.57204346,  -5.23015179,  0.54789149,  10.8714487 ]
        ]), 
        np.array([
            [ 0.62769453,  12.15734286,   9.36789219,  14.1546309, -12.25387748,   4.41250584,  12.19053681,  16.28144672, 11.25129557, -13.05971234],
            [ 0.16454356, -12.20169885,  -9.27708318, -13.93580189, 11.46380792,  -4.74118014, -12.24451267, -16.39697351,-11.31300638,  12.92324732]]
        )
    ]
    return theta_config

if __name__ == "__main__":

    # For XNOR dataset
    print('XNOR dataset')
    dataset = './datasets/xnor.csv'
    hidden_layers_conf = [2]
    theta_config = theta_config_for_xnor()
    to_predict = []
    to_predict.append(np.array([[0, 0]]).T)
    to_predict.append(np.array([[0, 1]]).T)
    to_predict.append(np.array([[1, 0]]).T)
    to_predict.append(np.array([[1, 1]]).T)
    
    run_for_dataset(dataset, hidden_layers_conf, theta_config, to_predict)
    
    # For blobs dataset
    print('blobs dataset')
    dataset = './datasets/blobs.csv'
    hidden_layers_conf = [2, 5]
    theta_config = theta_config_for_blobs()
    to_predict = []
    to_predict.append(np.array([[1, -9]]).T)
    
    to_predict.append(np.array([[-4, 7.8]]).T)
    to_predict.append(np.array([[-9, 4.5]]).T)
    run_for_dataset(dataset, hidden_layers_conf, theta_config, to_predict)

    # For moons dataset
    print('Moons dataset')
    dataset = './datasets/moons.csv'
    hidden_layers_conf = [4,4,4]
    theta_config = theta_config_for_moons()
    to_predict = []
    to_predict.append(np.array([[-0.5, 0.5]]).T)
    to_predict.append(np.array([[1, 0.5]]).T)
    to_predict.append(np.array([[0, 0]]).T)
    to_predict.append(np.array([[1.5, -0.5]]).T)
    run_for_dataset(dataset, hidden_layers_conf, theta_config, to_predict)

    # For circles dataset
    print('Circles dataset')
    dataset = './datasets/circles.csv'
    hidden_layers_conf = [9,9,9]
    theta_config = theta_config_for_circles()
    to_predict = []
    to_predict.append(np.array([[-0.6, -0.85]]).T)
    to_predict.append(np.array([[0.75, -0.06]]).T)
    run_for_dataset(dataset, hidden_layers_conf, theta_config, to_predict)

    plt.show()