# Method for unpickling probabilities/logits saved in process of evaluation.

import pickle

# Example of unpickle
FILE_PATH = '/root/autodl-tmp/HugCode/custom/calibration/logits/probs_defect-plbart_c10_logits.p'


# Open file with pickled variables
def unpickle_probs(file, verbose = 0):
    with open(file, 'rb') as f:  # Python 3: open(..., 'rb')
        (y_probs_val, y_val), (y_probs_test, y_test) = pickle.load(f)  # unpickle the content
        
    if verbose:    
        print("y_probs_val:", y_probs_val.shape)  # (5000, 10); Validation set probabilities of predictions
        print("y_true_val:", y_val.shape)  # (5000, 1); Validation set true labels
        print("y_probs_test:", y_probs_test.shape)  # (10000, 10); Test set probabilities
        print("y_true_test:", y_test.shape)  # (10000, 1); Test set true labels
        
    return ((y_probs_val, y_val), (y_probs_test, y_test))
    
    
if __name__ == '__main__':
    
    (y_probs_val, y_val), (y_probs_test, y_test) = unpickle_probs(FILE_PATH, True)
    print(type(y_probs_val))
    print(y_probs_val[:10])
    print(y_val[:10])

# y_probs_val: (5000, 10)
# y_true_val: (5000, 1)
# y_probs_test: (10000, 10)
# y_true_test: (10000, 1)
# [[ -4.3926244  -10.662071    15.001377     9.273258     0.9964444
#     3.1144302   -1.7024746   -3.281448    -7.9440007   -1.6698395 ]
#  [ 23.341518     2.2476478   -4.439763    -1.0728556   -0.15451592
#   -17.545488    -6.3943734   -5.2348266    4.7808843    3.2923217 ]
#  [ -5.7494907   -9.203811    -5.074207    22.31732      4.8800116
#    13.790758    -4.289096     0.689984    -9.400471    -9.517752  ]
#  [ -2.79247      7.6545734  -13.861414    -4.582863    -8.904366
#    -0.7214924   -7.074349     5.4982586  -13.808119    35.864895  ]
#  [-13.006293     2.3017666   -3.1628966   -6.8020644    2.3561826
#    -5.739541    46.786892   -10.448325   -13.664284    -0.8035959 ]
#  [-21.917038    -9.364584    -1.3682745   13.321063    12.610022
#    36.59152     -5.7101927    3.169985   -19.129087   -10.61969   ]
#  [ -0.8179345  -11.39099     31.377935    -2.6386137    5.3250947
#     6.040866   -15.475003   -10.960174    10.877698   -11.347913  ]
#  [-17.378632    -2.1381567   -3.6466894   14.621441    -2.9521701
#    22.423048    -0.3892364   -8.377205    -1.8670043   -1.8410743 ]
#  [-14.53798      8.608554   -10.925194    -4.211013     9.784155
#    -4.3383245  -11.656546    -6.9872346   -2.151785    33.090782  ]
#  [  9.521259    -0.7254843   -5.9856358   -4.1960406  -16.634945
#    -7.6551876  -14.377208    -3.29948     30.185848    12.344185  ]]
# [[2]
#  [0]
#  [3]
#  [9]
#  [6]
#  [5]
#  [2]
#  [3]
#  [9]
#  [8]]