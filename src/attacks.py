import numpy as np
import copy
import torch

def sign_flipping_attack(weights, attack_value = -1):
    weights_modified = copy.deepcopy(weights)
    for k in weights_modified.keys():
        weights_modified[k] = weights_modified[k] * attack_value     
    return weights_modified

def additive_noise(weights, args):
    weights_modified = copy.deepcopy(weights)
    for k in weights_modified.keys():
        noise = torch.from_numpy( copy.deepcopy( np.random.normal( scale=0.3, size=weights_modified[k].shape ) ) ).to(args.device) 
        weights_modified[k] = weights_modified[k] + noise     
    return weights_modified

if __name__ == "__main__":
    # test_input = np.random.rand(10, 10)
    test_input = np.array([ np.random.rand(10,1), np.random.rand(1, 10), np.random.rand(2,3,4), 
                    np.random.rand(5,4,3,2,2) ])
    print("[ORIGINAL]", test_input)
    # print([  same_value_attack(item) for item in test_input ])
    test_input = np.array([ random_attack(item) for item in test_input ])
    print(test_input)
    # print("[SAME VALUE]", same_value_attack(test_input))
    # print("[SIGN FLIP]", sign_flipping_attack(test_input))
    # print("[RANDOM]", random_attack(test_input))