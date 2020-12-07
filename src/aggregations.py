import copy
import torch
import numpy as np
import hdmedians as hdm
from scipy.stats import trim_mean
from sklearn.metrics import precision_recall_fscore_support
from models.Nets import VAE

def to_ndarray(w_locals):
    for user_idx in range(len(w_locals)): 
        for key in w_locals[user_idx]:
            w_locals[user_idx][key] = w_locals[user_idx][key].cpu().numpy()
    return w_locals

def reshape_from_oneD(one_d_vector, shape_size_dict, args=None):
	weight_dict = {}
	one_d_vector_idx = 0
	for key in shape_size_dict:
		weight_dict[key] =  copy.deepcopy( one_d_vector[one_d_vector_idx: (one_d_vector_idx + shape_size_dict[key][0])] ).reshape(shape_size_dict[key][1])

		if args != None:
			weight_dict[key] = torch.tensor(weight_dict[key]).to(args.device)

		one_d_vector_idx += shape_size_dict[key][0]
	return weight_dict

def aggregation(w_locals, user_weights, args, attacker_idx=None):

    layer_shape_size = {}
    for key in w_locals[0]:
        layer_shape_size[key] = ( w_locals[0][key].numel(), list(w_locals[0][key].shape) )
    print(layer_shape_size)

    # to numpy array
    w_locals = to_ndarray(w_locals)

    user_one_d = []
    for user_idx in range(len(w_locals)):
        tmp = np.array([])
        for key in w_locals[user_idx]:
            data_idx_key = np.array(w_locals[user_idx][key]).flatten()
            tmp = copy.deepcopy( np.hstack((tmp, data_idx_key)) )
        user_one_d.append(tmp)

    if args.aggregation == "Krum":
        print( "Using {} aggregation".format(args.aggregation) )
        num_machines = 100
        num_byz = 30
        score = []
        num_near = num_machines - num_byz - 2

        for i, w_i in enumerate(user_one_d):
            dist = []
            for j, w_j in enumerate(user_one_d):
                if i != j:
                    dist.append( np.linalg.norm(w_i - w_j) ** 2 )
            dist.sort(reverse=False)
            score.append(sum(dist[0:num_near]))
        i_star = score.index( min(score) )            

        selected = user_one_d[i_star]

        return reshape_from_oneD(selected, layer_shape_size, args)

    elif args.aggregation == "GeoMed":
        print( "Using {} aggregation".format(args.aggregation) )
        user_one_d = np.array(user_one_d).astype(float)

        selected = np.asarray(hdm.geomedian(user_one_d, axis=0))

        return reshape_from_oneD(selected, layer_shape_size, args)

    elif args.aggregation == "atten":
        # this is our method
        true = [False] * args.sample_users
        for idx in attacker_idx:
            true[idx] = True

        if args.dataset == "femnist":
            print("Loading femnist_index.npy...")
            femnist_index = np.load("femnist_index.npy")
            user_one_d_test = [ item[femnist_index] for item in copy.deepcopy(user_one_d) ]
        else:
            user_one_d_test = copy.deepcopy(user_one_d)

        model = VAE( input_dim = user_one_d_test[0].shape[0] )
        model.load_state_dict( torch.load(args.vae_model) )
        model.eval()

        scores = model.test(user_one_d_test)
        score_avg = np.mean(scores)
        print("scores", scores)
        print("score_avg", score_avg)
        pred = scores > score_avg
        print( "[Classification_p_r_f], {}".format( precision_recall_fscore_support(true, pred, average="binary") ) )

        new_weights = copy.deepcopy(user_weights)
        for _ in range(len(pred)):
            if pred[_]:
                new_weights[_] = 0.0
        new_weights = new_weights / sum(new_weights)

        user_one_d = np.array(user_one_d)
        selected = np.zeros(user_one_d[0].shape)
        for _ in range(len(new_weights)):
            selected += user_one_d[_] * new_weights[_]

        return reshape_from_oneD(selected, layer_shape_size, args)

    return

if __name__ == "__main__":

    test = {"a": np.array([[1,2],[3,4]]), "b": np.array([5,6,7,8]), "c": np.array([[[9,10], [11,12]], [[13,14], [15,16]]])}
    print(test)

    layer_shape_size = {}
    for key in test:
        layer_shape_size[key] = ( test[key].size, list(test[key].shape) )

    tmp = np.array([])
    for key in test:
        print(key)
        data_idx_key = np.array(test[key]).flatten()
        tmp = copy.deepcopy( np.hstack((tmp, data_idx_key)) )
    print(tmp)
    print(reshape_from_oneD(tmp, layer_shape_size))



