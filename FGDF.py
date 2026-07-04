
import warnings

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import tenseal as ts
import time
import torch
import math
import torch.nn.functional as F
import sys
import csv
from torch import nn
from torch.utils.data import TensorDataset
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
pd.options.display.float_format = "{:,.4f}".format
import random
import os

seed = 42

random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


dataset_name = 'YOUR DATA'
dataset_1 = pd.read_csv(dataset_name, header=None)
df = pd.DataFrame(dataset_1)
seed = 42
df = df.sample(frac=1, random_state=seed)
dataset_1 = df

num_of_features = (dataset_1.shape[1]) - 1
num_of_classes = 2

split_idx = math.floor(len(dataset_1) * 0.75)
train_DS = dataset_1[0:split_idx]
test_DS = dataset_1[split_idx:]

x_trainn = train_DS.iloc[:, 0:-1].values.astype(np.float32)
y_trainn = train_DS.iloc[:, -1].values.astype(np.int64)
x_test = test_DS.iloc[:, 0:-1].values.astype(np.float32)
y_test = test_DS.iloc[:, -1].values.astype(np.int64)

split_idx = math.floor(len(x_trainn) * 0.8)
x_train, y_train = x_trainn[:split_idx], y_trainn[:split_idx]
x_valid, y_valid = x_trainn[split_idx:], y_trainn[split_idx:]

def create_graph_data(x, y):
    graph_list = []
    n_samples = x.shape[0]
    for i in range(n_samples):
        x_node = torch.tensor(np.tile(x[i], (num_of_features, 1)), dtype=torch.float32)
        y_graph = torch.tensor([y[i]], dtype=torch.long)
        edge_index = build_sparse_edges(num_of_features, k=3)
        graph = Data(x=x_node, edge_index=edge_index, y=y_graph)
        graph_list.append(graph)
    return graph_list

def build_sparse_edges(n_nodes, k):
    edges = []
    for i in range(n_nodes):
        neighbors = [(j, abs(i-j)) for j in range(n_nodes) if i != j]
        neighbors.sort(key=lambda x: x[1])
        for j, _ in neighbors[:k]:
            edges.append([i, j])
            edges.append([j, i])
    return torch.tensor(edges, dtype=torch.long).T

def dividing_and_shuffling_labels(y_label, seed, amount):
    y_label = pd.DataFrame(y_label, columns=["labels"])
    y_label["i"] = np.arange(len(y_label))
    label_y_dict = dict()
    for i in range(2):
        var_name = "label" + str(i)
        label_info = y_label[y_label["labels"] == i]
        np.random.seed(seed)
        label_info = np.random.permutation(label_info)
        label_info = label_info[0:amount]
        label_info = pd.DataFrame(label_info, columns=["labels", "i"])
        label_y_dict.update({var_name: label_info})
    return label_y_dict

def get_subsamples(label_dict, number_of_samples, amount):
    sample_dict = dict()
    batch_size = int(math.floor(amount / number_of_samples))
    for i in range(number_of_samples):
        sample_name = "sample" + str(i)
        dumb = pd.DataFrame()
        for j in range(2):
            label_name = str("label") + str(j)
            a = label_dict[label_name][i * batch_size:(i + 1) * batch_size]
            dumb = pd.concat([dumb, a], axis=0)
        dumb.reset_index(drop=True, inplace=True)
        sample_dict.update({sample_name: dumb})
    return sample_dict

def create_graph_subsamples(sample_dict, x_data, y_data, x_name, y_name):
    graph_dict = dict()
    y_dict = dict()
    for i in range(len(sample_dict)):
        graph_name = x_name + str(i)
        yname = y_name + str(i)
        sample_name = "sample" + str(i)
        indices = np.sort(np.array(sample_dict[sample_name]["i"]))
        client_x = x_data[indices, :]
        client_y = y_data[indices]
        client_graphs = create_graph_data(client_x, client_y)
        graph_dict.update({graph_name: client_graphs})
        y_dict.update({yname: client_y})
    return graph_dict, y_dict

class GATNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        from torch_geometric.nn import global_mean_pool
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x


class enc_model_weight:
    def __init__(self, in_channels, hidden_channels, out_channels):
        self.conv1_enc = None
        self.conv2_enc = None
        self.conv3_enc = None
        self.fc_enc = None

        self.conv1_enc_B = None
        self.conv2_enc_B = None
        self.conv3_enc_B = None
        self.fc_enc_B = None

class dec_model_weight:
    def __init__(self, in_channels, hidden_channels, out_channels):
        self.conv1_dec = None
        self.conv2_dec = None
        self.conv3_dec = None
        self.fc_dec = None
        self.conv1_dec_B = None
        self.conv2_dec_B = None
        self.conv3_dec_B = None
        self.fc_dec_B = None


def train(model, train_loader, criterion, optimizer):
    model.train()
    train_loss = 0.0
    correct = 0
    for data in train_loader:
        output = model(data)
        loss = criterion(output, data.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        prediction = output.argmax(dim=1, keepdim=True)
        correct += prediction.eq(data.y.view_as(prediction)).sum().item()
    train_loss /= len(train_loader)
    correct /= len(train_loader.dataset)
    return train_loss, correct

def validation(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    recall_all = 0
    precision_all = 0
    f1_score_all = 0
    TN_all, FP_all, FN_all, TP_all = 0, 0, 0, 0
    with torch.no_grad():
        for data in test_loader:
            output = model(data)
            test_loss += criterion(output, data.y).item()
            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(data.y.view_as(prediction)).sum().item()
            y_true = data.y.cpu().numpy()
            y_pred = prediction.cpu().numpy().flatten()
            recall = recall_score(y_true, y_pred, zero_division=0)
            precision = precision_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            recall_all += recall
            precision_all += precision
            f1_score_all += f1
            CM = confusion_matrix(y_true, y_pred, labels=[0, 1])
            if CM.shape == (2, 2):
                TN, FP, FN, TP = CM[0][0], CM[0][1], CM[1][0], CM[1][1]
            else:
                TN, FP, FN, TP = 0, 0, 0, 0
            TN_all += TN
            FP_all += FP
            FN_all += FN
            TP_all += TP
    test_loss /= len(test_loader)
    correct /= len(test_loader.dataset)
    recall_all /= len(test_loader)
    precision_all /= len(test_loader)
    f1_score_all /= len(test_loader)
    return test_loss, correct, recall_all, precision_all, f1_score_all, TN_all, FP_all, FN_all, TP_all, len(test_loader)

def create_model_optimizer_criterion_dict(number_of_samples, in_channels, hidden_channels, out_channels):
    model_dict = {}
    optimizer_dict = {}
    criterion_dict = {}
    encrypted_model_dict = {}
    decrypted_model_dict = {}
    for i in range(number_of_samples):
        model_name = "model" + str(i)
        model_info = GATNet(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels)
        model_dict[model_name] = model_info
        enc_m_name = "enc_model" + str(i)
        enc_m_info = enc_model_weight(in_channels, hidden_channels, out_channels)
        encrypted_model_dict[enc_m_name] = enc_m_info
        dec_m_name = "dec_model" + str(i)
        dec_m_info = dec_model_weight(in_channels, hidden_channels, out_channels)
        decrypted_model_dict[dec_m_name] = dec_m_info
        optimizer_name = "optimizer" + str(i)
        optimizer_info = torch.optim.SGD(model_info.parameters(), lr=learning_rate, momentum=momentum)
        optimizer_dict[optimizer_name] = optimizer_info
        criterion_name = "criterion" + str(i)
        criterion_info = nn.CrossEntropyLoss()
        criterion_dict[criterion_name] = criterion_info
    return model_dict, optimizer_dict, criterion_dict, encrypted_model_dict, decrypted_model_dict


def decrypt(enc):

    values = enc.decrypt()

    if hasattr(values, "tolist"):
        return values.tolist()

    return values


def context():
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    ctx.global_scale = pow(2, 40)
    ctx.generate_galois_keys()
    return ctx

context = context()

def apply_FHE(v1):

    flat_values = (
        v1.detach()
        .cpu()
        .float()
        .reshape(-1)
        .tolist()
    )

    encrypted_vector = ts.ckks_vector(
        context,
        flat_values
    )

    return encrypted_vector



def enc_model(model_dict, encrypted_model_dict, number_of_samples, epsilon, delta, sensitivity, use_dp_noise=False):
    with torch.no_grad():
        for i in range(number_of_samples):


            w1 = model_dict[name_of_models[i]].conv1.lin.weight.data.clone()
            w1_to_encrypt = add_gaussian_noise(w1, epsilon / 2, delta, sensitivity) if use_dp_noise else w1
            encrypted_model_dict[name_of_enc_models[i]].conv1_enc = apply_FHE(w1_to_encrypt)



            if i == 0:
                enc_w1 = encrypted_model_dict[name_of_enc_models[i]].conv1_enc

            encrypted_model_dict[name_of_enc_models[i]].conv1_enc_B = model_dict[name_of_models[i]].conv1.bias.data.clone()


            w2 = model_dict[name_of_models[i]].conv2.lin.weight.data.clone()
            w2_to_encrypt = add_gaussian_noise(w2, epsilon / 2, delta, sensitivity) if use_dp_noise else w2
            encrypted_model_dict[name_of_enc_models[i]].conv2_enc = apply_FHE(w2_to_encrypt)
            encrypted_model_dict[name_of_enc_models[i]].conv2_enc_B = model_dict[name_of_models[i]].conv2.bias.data.clone()


            w3 = model_dict[name_of_models[i]].conv3.lin.weight.data.clone()
            w3_to_encrypt = add_gaussian_noise(w3, epsilon / 2, delta, sensitivity) if use_dp_noise else w3
            encrypted_model_dict[name_of_enc_models[i]].conv3_enc = apply_FHE(w3_to_encrypt)
            encrypted_model_dict[name_of_enc_models[i]].conv3_enc_B = model_dict[name_of_models[i]].conv3.bias.data.clone()


            w_fc = model_dict[name_of_models[i]].fc.weight.data.clone()
            w_fc_to_encrypt = add_gaussian_noise(w_fc, epsilon / 2, delta, sensitivity) if use_dp_noise else w_fc
            encrypted_model_dict[name_of_enc_models[i]].fc_enc = apply_FHE(w_fc_to_encrypt)
            encrypted_model_dict[name_of_enc_models[i]].fc_enc_B = model_dict[name_of_models[i]].fc.bias.data.clone()

    return encrypted_model_dict


def dec_model(
    encrypted_model_dict,
    decrypted_model_dict,
    number_of_samples
):
    with torch.no_grad():
        for i in range(number_of_samples):


            conv1_shape = tuple(
                model_dict[name_of_models[i]]
                .conv1.lin.weight.shape
            )

            conv2_shape = tuple(
                model_dict[name_of_models[i]]
                .conv2.lin.weight.shape
            )

            conv3_shape = tuple(
                model_dict[name_of_models[i]]
                .conv3.lin.weight.shape
            )

            fc_shape = tuple(
                model_dict[name_of_models[i]]
                .fc.weight.shape
            )


            conv1_dec_w = torch.tensor(
                decrypt(
                    encrypted_model_dict[
                        name_of_enc_models[i]
                    ].conv1_enc
                ),
                dtype=torch.float32
            ).reshape(conv1_shape)

            conv2_dec_w = torch.tensor(
                decrypt(
                    encrypted_model_dict[
                        name_of_enc_models[i]
                    ].conv2_enc
                ),
                dtype=torch.float32
            ).reshape(conv2_shape)

            conv3_dec_w = torch.tensor(
                decrypt(
                    encrypted_model_dict[
                        name_of_enc_models[i]
                    ].conv3_enc
                ),
                dtype=torch.float32
            ).reshape(conv3_shape)

            fc_dec_w = torch.tensor(
                decrypt(
                    encrypted_model_dict[
                        name_of_enc_models[i]
                    ].fc_enc
                ),
                dtype=torch.float32
            ).reshape(fc_shape)


            decrypted_model_dict[
                name_of_dec_models[i]
            ].conv1_dec = conv1_dec_w.tolist()

            decrypted_model_dict[
                name_of_dec_models[i]
            ].conv2_dec = conv2_dec_w.tolist()

            decrypted_model_dict[
                name_of_dec_models[i]
            ].conv3_dec = conv3_dec_w.tolist()

            decrypted_model_dict[
                name_of_dec_models[i]
            ].fc_dec = fc_dec_w.tolist()


            decrypted_model_dict[
                name_of_dec_models[i]
            ].conv1_dec_B = encrypted_model_dict[
                name_of_enc_models[i]
            ].conv1_enc_B

            decrypted_model_dict[
                name_of_dec_models[i]
            ].conv2_dec_B = encrypted_model_dict[
                name_of_enc_models[i]
            ].conv2_enc_B

            decrypted_model_dict[
                name_of_dec_models[i]
            ].conv3_dec_B = encrypted_model_dict[
                name_of_enc_models[i]
            ].conv3_enc_B

            decrypted_model_dict[
                name_of_dec_models[i]
            ].fc_dec_B = encrypted_model_dict[
                name_of_enc_models[i]
            ].fc_enc_B

    return decrypted_model_dict



def tensor_bytes(tensor):

    return tensor.numel() * tensor.element_size()


def get_client_upload_bytes(user):

    encrypted_weight_bytes = (
        len(user.conv1_enc.serialize())
        + len(user.conv2_enc.serialize())
        + len(user.conv3_enc.serialize())
        + len(user.fc_enc.serialize())
    )

    plaintext_bias_bytes = (
        tensor_bytes(user.conv1_enc_B)
        + tensor_bytes(user.conv2_enc_B)
        + tensor_bytes(user.conv3_enc_B)
        + tensor_bytes(user.fc_enc_B)
    )

    return encrypted_weight_bytes + plaintext_bias_bytes


def Server_get_averaged_weights(
    matrix_dict,
    number_of_samples,
    epsilon,
    delta,
    sensitivity,
    context,
    use_dp_noise=False
):


    conv1_shape = tuple(
        model_dict[name_of_models[0]].conv1.lin.weight.shape
    )
    conv2_shape = tuple(
        model_dict[name_of_models[0]].conv2.lin.weight.shape
    )
    conv3_shape = tuple(
        model_dict[name_of_models[0]].conv3.lin.weight.shape
    )
    fc_shape = tuple(
        model_dict[name_of_models[0]].fc.weight.shape
    )

    first_user = matrix_dict["user_0"]


    conv1_mean_weight = first_user.conv1_enc.copy()
    conv2_mean_weight = first_user.conv2_enc.copy()
    conv3_mean_weight = first_user.conv3_enc.copy()
    fc_mean_weight = first_user.fc_enc.copy()


    conv1_mean_bias = first_user.conv1_enc_B.clone()
    conv2_mean_bias = first_user.conv2_enc_B.clone()
    conv3_mean_bias = first_user.conv3_enc_B.clone()
    fc_mean_bias = first_user.fc_enc_B.clone()


    total_size_Enc_W_B = get_client_upload_bytes(first_user)

    with torch.no_grad():

        for i in range(1, number_of_samples):
            user = matrix_dict[f"user_{i}"]

            conv1_mean_weight += user.conv1_enc
            conv2_mean_weight += user.conv2_enc
            conv3_mean_weight += user.conv3_enc
            fc_mean_weight += user.fc_enc

            conv1_mean_bias += user.conv1_enc_B
            conv2_mean_bias += user.conv2_enc_B
            conv3_mean_bias += user.conv3_enc_B
            fc_mean_bias += user.fc_enc_B

            total_size_Enc_W_B += get_client_upload_bytes(user)


        average_factor = 1.0 / number_of_samples

        conv1_mean_weight *= average_factor
        conv2_mean_weight *= average_factor
        conv3_mean_weight *= average_factor
        fc_mean_weight *= average_factor

        conv1_mean_bias /= number_of_samples
        conv2_mean_bias /= number_of_samples
        conv3_mean_bias /= number_of_samples
        fc_mean_bias /= number_of_samples


        if use_dp_noise:
            conv1_noise = generate_encrypted_gaussian_noise(
                shape=conv1_shape,
                epsilon=epsilon / 2,
                delta=delta,
                sensitivity=sensitivity / number_of_samples,
                context=context
            )

            conv2_noise = generate_encrypted_gaussian_noise(
                shape=conv2_shape,
                epsilon=epsilon / 2,
                delta=delta,
                sensitivity=sensitivity / number_of_samples,
                context=context
            )

            conv3_noise = generate_encrypted_gaussian_noise(
                shape=conv3_shape,
                epsilon=epsilon / 2,
                delta=delta,
                sensitivity=sensitivity / number_of_samples,
                context=context
            )

            fc_noise = generate_encrypted_gaussian_noise(
                shape=fc_shape,
                epsilon=epsilon / 2,
                delta=delta,
                sensitivity=sensitivity / number_of_samples,
                context=context
            )

            conv1_mean_weight += conv1_noise
            conv2_mean_weight += conv2_noise
            conv3_mean_weight += conv3_noise
            fc_mean_weight += fc_noise

    return (
        conv1_mean_weight,
        conv1_mean_bias,
        conv2_mean_weight,
        conv2_mean_bias,
        conv3_mean_weight,
        conv3_mean_bias,
        fc_mean_weight,
        fc_mean_bias,
        total_size_Enc_W_B
    )

def update_dec_main_model(dec_main_model, decrypted_model_dict, number_of_samples):
    with torch.no_grad():
        dec_main_model.conv1_dec = torch.zeros_like(torch.tensor(decrypted_model_dict['dec_model0'].conv1_dec))
        dec_main_model.conv2_dec = torch.zeros_like(torch.tensor(decrypted_model_dict['dec_model0'].conv2_dec))
        dec_main_model.conv3_dec = torch.zeros_like(torch.tensor(decrypted_model_dict['dec_model0'].conv3_dec))
        dec_main_model.fc_dec = torch.zeros_like(torch.tensor(decrypted_model_dict['dec_model0'].fc_dec))
        dec_main_model.conv1_dec_B = torch.zeros_like(decrypted_model_dict['dec_model0'].conv1_dec_B)
        dec_main_model.conv2_dec_B = torch.zeros_like(decrypted_model_dict['dec_model0'].conv2_dec_B)
        dec_main_model.conv3_dec_B = torch.zeros_like(decrypted_model_dict['dec_model0'].conv3_dec_B)
        dec_main_model.fc_dec_B = torch.zeros_like(decrypted_model_dict['dec_model0'].fc_dec_B)
        for i in range(number_of_samples):
            dec_main_model.conv1_dec += torch.tensor(decrypted_model_dict[name_of_dec_models[i]].conv1_dec)
            dec_main_model.conv2_dec += torch.tensor(decrypted_model_dict[name_of_dec_models[i]].conv2_dec)
            dec_main_model.conv3_dec += torch.tensor(decrypted_model_dict[name_of_dec_models[i]].conv3_dec)
            dec_main_model.fc_dec += torch.tensor(decrypted_model_dict[name_of_dec_models[i]].fc_dec)
            dec_main_model.conv1_dec_B += decrypted_model_dict[name_of_dec_models[i]].conv1_dec_B
            dec_main_model.conv2_dec_B += decrypted_model_dict[name_of_dec_models[i]].conv2_dec_B
            dec_main_model.conv3_dec_B += decrypted_model_dict[name_of_dec_models[i]].conv3_dec_B
            dec_main_model.fc_dec_B += decrypted_model_dict[name_of_dec_models[i]].fc_dec_B
        dec_main_model.conv1_dec /= number_of_samples
        dec_main_model.conv2_dec /= number_of_samples
        dec_main_model.conv3_dec /= number_of_samples
        dec_main_model.fc_dec /= number_of_samples
        dec_main_model.conv1_dec_B /= number_of_samples
        dec_main_model.conv2_dec_B /= number_of_samples
        dec_main_model.conv3_dec_B /= number_of_samples
        dec_main_model.fc_dec_B /= number_of_samples
    return dec_main_model

def copy_dec_main_model_to_main_model(main_model, dec_main_model):
    with torch.no_grad():
        main_model.conv1.lin.weight.data = torch.tensor(dec_main_model.conv1_dec).clone().detach()
        main_model.conv2.lin.weight.data = torch.tensor(dec_main_model.conv2_dec).clone().detach()
        main_model.conv3.lin.weight.data = torch.tensor(dec_main_model.conv3_dec).clone().detach()
        main_model.fc.weight.data = torch.tensor(dec_main_model.fc_dec).clone().detach()
        main_model.conv1.bias.data = torch.tensor(dec_main_model.conv1_dec_B).clone().detach()
        main_model.conv2.bias.data = torch.tensor(dec_main_model.conv2_dec_B).clone().detach()
        main_model.conv3.bias.data = torch.tensor(dec_main_model.conv3_dec_B).clone().detach()
        main_model.fc.bias.data = torch.tensor(dec_main_model.fc_dec_B).clone().detach()
    return main_model

def set_averaged_Enc_weights_as_main_Enc_model_weights_and_update_main_Enc_model(enc_main_model, conv1_mean_weight, conv2_mean_weight, conv3_mean_weight, fc_mean_weight, conv1_mean_bias, conv2_mean_bias, conv3_mean_bias, fc_mean_bias):
    with torch.no_grad():
        enc_main_model.conv1_enc = conv1_mean_weight
        enc_main_model.conv2_enc = conv2_mean_weight
        enc_main_model.conv3_enc = conv3_mean_weight
        enc_main_model.fc_enc = fc_mean_weight
        enc_main_model.conv1_enc_B = conv1_mean_bias
        enc_main_model.conv2_enc_B = conv2_mean_bias
        enc_main_model.conv3_enc_B = conv3_mean_bias
        enc_main_model.fc_enc_B = fc_mean_bias
    return enc_main_model

def send_main_model_to_nodes_and_update_model_dict_before_encryption(main_model, model_dict, number_of_samples):
    with torch.no_grad():
        for i in range(number_of_samples):
            model_dict[name_of_models[i]].conv1.lin.weight.data = main_model.conv1.lin.weight.data.clone()
            model_dict[name_of_models[i]].conv2.lin.weight.data = main_model.conv2.lin.weight.data.clone()
            model_dict[name_of_models[i]].conv3.lin.weight.data = main_model.conv3.lin.weight.data.clone()
            model_dict[name_of_models[i]].fc.weight.data = main_model.fc.weight.data.clone()
            model_dict[name_of_models[i]].conv1.bias.data = main_model.conv1.bias.data.clone()
            model_dict[name_of_models[i]].conv2.bias.data = main_model.conv2.bias.data.clone()
            model_dict[name_of_models[i]].conv3.bias.data = main_model.conv3.bias.data.clone()
            model_dict[name_of_models[i]].fc.bias.data = main_model.fc.bias.data.clone()
    return model_dict

def send_Enc_model_to_nodes_and_update_Enc_model_dict(enc_main_model, encrypted_model_dict, number_of_samples):
    with torch.no_grad():
        for i in range(number_of_samples):
            encrypted_model_dict[name_of_enc_models[i]].conv1_enc = enc_main_model.conv1_enc
            encrypted_model_dict[name_of_enc_models[i]].conv2_enc = enc_main_model.conv2_enc
            encrypted_model_dict[name_of_enc_models[i]].conv3_enc = enc_main_model.conv3_enc
            encrypted_model_dict[name_of_enc_models[i]].fc_enc = enc_main_model.fc_enc
            encrypted_model_dict[name_of_enc_models[i]].conv1_enc_B = enc_main_model.conv1_enc_B
            encrypted_model_dict[name_of_enc_models[i]].conv2_enc_B = enc_main_model.conv2_enc_B
            encrypted_model_dict[name_of_enc_models[i]].conv3_enc_B = enc_main_model.conv3_enc_B
            encrypted_model_dict[name_of_enc_models[i]].fc_enc_B = enc_main_model.fc_enc_B
    return encrypted_model_dict

def send_Enc_model_to_Batching_Matrix_updates(matrix_dict, encrypted_model_dict, number_of_samples):
    with torch.no_grad():
        for i in range(number_of_samples):
            matrix_dict["user_" + str(i)].conv1_enc = encrypted_model_dict[name_of_enc_models[i]].conv1_enc
            matrix_dict["user_" + str(i)].conv2_enc = encrypted_model_dict[name_of_enc_models[i]].conv2_enc
            matrix_dict["user_" + str(i)].conv3_enc = encrypted_model_dict[name_of_enc_models[i]].conv3_enc
            matrix_dict["user_" + str(i)].fc_enc = encrypted_model_dict[name_of_enc_models[i]].fc_enc
            matrix_dict["user_" + str(i)].conv1_enc_B = encrypted_model_dict[name_of_enc_models[i]].conv1_enc_B
            matrix_dict["user_" + str(i)].conv2_enc_B = encrypted_model_dict[name_of_enc_models[i]].conv2_enc_B
            matrix_dict["user_" + str(i)].conv3_enc_B = encrypted_model_dict[name_of_enc_models[i]].conv3_enc_B
            matrix_dict["user_" + str(i)].fc_enc_B = encrypted_model_dict[name_of_enc_models[i]].fc_enc_B
    return matrix_dict

def start_train_end_node_process_without_print(number_of_samples, model_dict):
    for i in range(number_of_samples):
        train_graphs = train_graph_dict[name_of_x_train_sets[i]]
        test_graphs = test_graph_dict[name_of_x_test_sets[i]]
        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_graphs, batch_size=batch_size * 2)
        model = model_dict[name_of_models[i]]
        criterion = criterion_dict[name_of_criterions[i]]
        optimizer = optimizer_dict[name_of_optimizers[i]]
        for epoch in range(local_epochs):
            train_loss, train_accuracy = train(model, train_loader, criterion, optimizer)
            test_loss, test_accuracy, recall_all, precision_all, f1_score_all, TN_all, FP_all, FN_all, TP_all, len_test_loader = validation(model, test_loader, criterion)

    return model_dict

def create_matrix_for_users(number_of_samples, in_channels, hidden_channels, out_channels):
    matrix_dict_ = {}
    for i in range(number_of_samples):
        matrix_name_user = "user_" + str(i)
        matrix_info_user = enc_model_weight(in_channels, hidden_channels, out_channels)
        matrix_dict_[matrix_name_user] = matrix_info_user
    return matrix_dict_


batch_size = 32
learning_rate = 0.01
numEpoch = 30
local_epochs=30
momentum = 0.9
number_of_samples = 100
number_of_clusters = 5

in_channels = num_of_features
hidden_channels = 16
out_channels = num_of_classes

train_amount = 6000
valid_amount = 1000
test_amount = 2000
print_amount = 3


label_dict_train = dividing_and_shuffling_labels(y_label=y_train, seed=1, amount=train_amount)
sample_dict_train = get_subsamples(label_dict=label_dict_train, number_of_samples=number_of_samples, amount=train_amount)
train_graph_dict, y_train_dict = create_graph_subsamples(sample_dict=sample_dict_train, x_data=x_train, y_data=y_train, x_name="x_train", y_name="y_train")

label_dict_valid = dividing_and_shuffling_labels(y_label=y_valid, seed=1, amount=valid_amount)
sample_dict_valid = get_subsamples(label_dict=label_dict_valid, number_of_samples=number_of_samples, amount=valid_amount)
valid_graph_dict, y_valid_dict = create_graph_subsamples(sample_dict=sample_dict_valid, x_data=x_valid, y_data=y_valid, x_name="x_valid", y_name="y_valid")

label_dict_test = dividing_and_shuffling_labels(y_label=y_test, seed=1, amount=test_amount)
sample_dict_test = get_subsamples(label_dict=label_dict_test, number_of_samples=number_of_samples, amount=test_amount)
test_graph_dict, y_test_dict = create_graph_subsamples(sample_dict=sample_dict_test, x_data=x_test, y_data=y_test, x_name="x_test", y_name="y_test")

main_model = GATNet(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels)


main_optimizer = torch.optim.SGD(main_model.parameters(), lr=learning_rate, momentum=momentum)
main_criterion = nn.CrossEntropyLoss()

enc_main_model = enc_model_weight(in_channels, hidden_channels, out_channels)
dec_main_model = dec_model_weight(in_channels, hidden_channels, out_channels)

model_dict, optimizer_dict, criterion_dict, encrypted_model_dict, decrypted_model_dict = create_model_optimizer_criterion_dict(
    number_of_samples=number_of_samples, in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels)

name_of_x_train_sets = list(train_graph_dict.keys())
name_of_y_train_sets = list(y_train_dict.keys())
name_of_x_valid_sets = list(valid_graph_dict.keys())
name_of_y_valid_sets = list(y_valid_dict.keys())
name_of_x_test_sets = list(test_graph_dict.keys())
name_of_y_test_sets = list(y_test_dict.keys())
name_of_models = list(model_dict.keys())
name_of_optimizers = list(optimizer_dict.keys())
name_of_criterions = list(criterion_dict.keys())
name_of_enc_models = list(encrypted_model_dict.keys())
name_of_dec_models = list(decrypted_model_dict.keys())
# -------------------------- 全局测试集数据加载器（GAT专用） --------------------------
global_test_graphs = create_graph_data(x_test, y_test)
test_dl = DataLoader(global_test_graphs, batch_size=batch_size * 2)

def stratified_member_nonmember_sample(
    member_indices,
    member_y_all,
    nonmember_indices,
    nonmember_y_all,
    max_per_class=1000,
    seed=42
):
    rng = np.random.default_rng(seed)

    member_indices = np.asarray(member_indices, dtype=int)
    nonmember_indices = np.asarray(nonmember_indices, dtype=int)

    member_selected = []
    nonmember_selected = []

    classes = np.unique(np.concatenate([member_y_all, nonmember_y_all]))

    for c in classes:
        member_c = member_indices[member_y_all[member_indices] == c]
        nonmember_c = nonmember_indices[nonmember_y_all[nonmember_indices] == c]

        n_c = min(len(member_c), len(nonmember_c), max_per_class)

        if n_c == 0:
            continue

        member_selected.extend(rng.choice(member_c, size=n_c, replace=False))
        nonmember_selected.extend(rng.choice(nonmember_c, size=n_c, replace=False))

    return np.array(member_selected, dtype=int), np.array(nonmember_selected, dtype=int)


member_indices_all = []
for i in range(number_of_samples):
    member_indices_all.extend(sample_dict_train[f"sample{i}"]["i"].tolist())

member_indices_all = np.array(sorted(list(set(member_indices_all))))


nonmember_indices_all = np.arange(len(x_test))

privacy_eval_size = min(len(member_indices_all), len(nonmember_indices_all), 2000)

rng = np.random.default_rng(42)
member_sel = rng.choice(member_indices_all, size=privacy_eval_size, replace=False)
nonmember_sel = rng.choice(nonmember_indices_all, size=privacy_eval_size, replace=False)

member_x = x_train[member_sel]
member_y = y_train[member_sel]

nonmember_x = x_test[nonmember_sel]
nonmember_y = y_test[nonmember_sel]

member_graphs = create_graph_data(member_x, member_y)
nonmember_graphs = create_graph_data(nonmember_x, nonmember_y)

member_dl = DataLoader(member_graphs, batch_size=batch_size * 2, shuffle=False)
nonmember_dl = DataLoader(nonmember_graphs, batch_size=batch_size * 2, shuffle=False)


epsilon = 1
delta = 1e-5
sensitivity = 0.1

use_dp_noise =False

def add_gaussian_noise(param, epsilon, delta, sensitivity):
    sigma = torch.sqrt(torch.tensor(2 * torch.log(torch.tensor(1.25 / delta))) * (sensitivity ** 2) / (epsilon ** 2))
    noise = torch.normal(mean=0.0, std=sigma.item(), size=param.shape)
    return param + noise

def generate_encrypted_gaussian_noise(
    shape,
    epsilon,
    delta,
    sensitivity,
    context
):
    sigma = math.sqrt(
        2.0
        * math.log(1.25 / delta)
        * (sensitivity ** 2)
        / (epsilon ** 2)
    )

    noise = torch.normal(
        mean=0.0,
        std=sigma,
        size=tuple(shape)
    )


    return ts.ckks_vector(
        context,
        noise.reshape(-1).tolist()
    )


def get_gaussian_sigma(epsilon, delta, sensitivity):
    return torch.sqrt(torch.tensor(2 * torch.log(torch.tensor(1.25 / delta))) * (sensitivity ** 2) / (epsilon ** 2))


def get_ciphertext_bytes(enc_obj):

    return len(enc_obj.serialize())
def get_tensor_bytes(tensor_obj):

    return tensor_obj.numel() * tensor_obj.element_size()

def safe_softmax_probs(logits):

    probs = F.softmax(logits, dim=1)
    probs = torch.clamp(probs, min=1e-12, max=1.0)
    probs = probs / probs.sum(dim=1, keepdim=True)
    return probs


def collect_privacy_features(model, loader):

    model.eval()
    feature_list = []
    prob_list = []

    with torch.no_grad():
        for data in loader:
            logits = model(data)
            probs = safe_softmax_probs(logits)
            y_true = data.y.view(-1)


            true_class_prob = probs[torch.arange(len(y_true)), y_true]


            max_prob, _ = probs.max(dim=1)


            entropy_val = -torch.sum(probs * torch.log(probs), dim=1)


            top2_vals, _ = torch.topk(probs, k=min(2, probs.shape[1]), dim=1)
            if probs.shape[1] >= 2:
                margin = top2_vals[:, 0] - top2_vals[:, 1]
            else:
                margin = top2_vals[:, 0]


            neg_loss = torch.log(true_class_prob)


            batch_features = torch.stack([
                true_class_prob,
                max_prob,
                entropy_val,
                margin,
                neg_loss
            ], dim=1)

            feature_list.append(batch_features.cpu().numpy())
            prob_list.append(probs.cpu().numpy())

    features = np.vstack(feature_list)
    all_probs = np.vstack(prob_list)
    avg_prob = np.mean(all_probs, axis=0)

    return features, avg_prob


def compute_kl_divergence(p, q, eps=1e-12):

    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)

    p = p / np.sum(p)
    q = q / np.sum(q)

    return np.sum(p * np.log(p / q))


def compute_bidirectional_kl(p, q, eps=1e-12):

    kl_pq = compute_kl_divergence(p, q, eps)
    kl_qp = compute_kl_divergence(q, p, eps)
    return 0.5 * (kl_pq + kl_qp), kl_pq, kl_qp


def compute_privacy_metrics(model, member_loader, nonmember_loader, random_state=42):

    member_features, member_avg_prob = collect_privacy_features(model, member_loader)
    nonmember_features, nonmember_avg_prob = collect_privacy_features(model, nonmember_loader)


    n = min(len(member_features), len(nonmember_features))
    member_features = member_features[:n]
    nonmember_features = nonmember_features[:n]

    X = np.vstack([member_features, nonmember_features])
    y = np.concatenate([
        np.ones(n, dtype=int),
        np.zeros(n, dtype=int)
    ])


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    mi_scores = mutual_info_classif(
        X_scaled,
        y,
        discrete_features=False,
        random_state=random_state
    )

    mi_mean = float(np.mean(mi_scores))
    mi_max = float(np.max(mi_scores))

    kl_sym, kl_mn, kl_nm = compute_bidirectional_kl(member_avg_prob, nonmember_avg_prob)

    return {
        "mi_mean": mi_mean,
        "mi_max": mi_max,
        "kl_sym": float(kl_sym),
        "kl_member_nonmember": float(kl_mn),
        "kl_nonmember_member": float(kl_nm)
    }
# ----------------- 主循环 -----------------
ExportToFile = 'FGDF' + dataset_name
Export = True
Flag = False

total_upload_all_rounds = 0
total_download_all_rounds = 0

for round_idx in range(numEpoch):
    start_time = time.time()

    model_dict = send_main_model_to_nodes_and_update_model_dict_before_encryption(main_model, model_dict, number_of_samples)


    local_start = time.perf_counter()

    model_dict = start_train_end_node_process_without_print(number_of_samples, model_dict)
    local_end = time.perf_counter()


    ENC_start = time.perf_counter()

    encrypted_model_dict = enc_model(
        model_dict, encrypted_model_dict, number_of_samples, epsilon, delta, sensitivity,
        use_dp_noise=use_dp_noise)
    ENC_end = time.perf_counter()


    total_upload_bytes_round = 0
    upload_bytes_each_client = []

    for client_idx in range(number_of_samples):
        enc = encrypted_model_dict[name_of_enc_models[client_idx]]


        encrypted_weight_bytes = (
                get_ciphertext_bytes(enc.conv1_enc)
                + get_ciphertext_bytes(enc.conv2_enc)
                + get_ciphertext_bytes(enc.conv3_enc)
                + get_ciphertext_bytes(enc.fc_enc)
        )


        plaintext_bias_bytes = (
                get_tensor_bytes(enc.conv1_enc_B)
                + get_tensor_bytes(enc.conv2_enc_B)
                + get_tensor_bytes(enc.conv3_enc_B)
                + get_tensor_bytes(enc.fc_enc_B)
        )

        client_upload_bytes = (
                encrypted_weight_bytes
                + plaintext_bias_bytes
        )

        upload_bytes_each_client.append(client_upload_bytes)
        total_upload_bytes_round += client_upload_bytes

    upload_bytes_per_client = (
            sum(upload_bytes_each_client)
            / len(upload_bytes_each_client)
    )

    upload_mb_per_client = upload_bytes_per_client / (1024 ** 2)
    upload_mb_round = total_upload_bytes_round / (1024 ** 2)


    matrix_dict_ = create_matrix_for_users(number_of_samples, in_channels, hidden_channels, out_channels)
    matrix_dict_ = send_Enc_model_to_Batching_Matrix_updates(matrix_dict_, encrypted_model_dict, number_of_samples)


    sever_start = time.perf_counter()

    (
        conv1_mean_weight,
        conv1_mean_bias,
        conv2_mean_weight,
        conv2_mean_bias,
        conv3_mean_weight,
        conv3_mean_bias,
        fc_mean_weight,
        fc_mean_bias,
        total_size_Enc_W_B
    ) = Server_get_averaged_weights(
        matrix_dict_,
        number_of_samples,
        epsilon,
        delta,
        sensitivity,
        context,
        use_dp_noise=use_dp_noise
    )
    sever_end = time.perf_counter()



    down_weight_bytes_per_client = (
            get_ciphertext_bytes(conv1_mean_weight)
            + get_ciphertext_bytes(conv2_mean_weight)
            + get_ciphertext_bytes(conv3_mean_weight)
            + get_ciphertext_bytes(fc_mean_weight)
    )

    down_bias_bytes_per_client = (
            get_tensor_bytes(conv1_mean_bias)
            + get_tensor_bytes(conv2_mean_bias)
            + get_tensor_bytes(conv3_mean_bias)
            + get_tensor_bytes(fc_mean_bias)
    )

    down_bytes_per_client = (
            down_weight_bytes_per_client
            + down_bias_bytes_per_client
    )

    total_download_bytes_round = (
            down_bytes_per_client
            * number_of_samples
    )

    download_mb_per_client = down_bytes_per_client / (1024 ** 2)
    download_mb_round = total_download_bytes_round / (1024 ** 2)



    total_upload_all_rounds += total_upload_bytes_round
    total_download_all_rounds += total_download_bytes_round


    enc_main_model = set_averaged_Enc_weights_as_main_Enc_model_weights_and_update_main_Enc_model(
        enc_main_model, conv1_mean_weight, conv2_mean_weight, conv3_mean_weight, fc_mean_weight,
        conv1_mean_bias, conv2_mean_bias, conv3_mean_bias, fc_mean_bias)


    encrypted_model_dict = send_Enc_model_to_nodes_and_update_Enc_model_dict(enc_main_model, encrypted_model_dict, number_of_samples)

    DEC_start = time.perf_counter()

    decrypted_model_dict = dec_model(encrypted_model_dict, decrypted_model_dict, number_of_samples)
    DEC_end = time.perf_counter()


    dec_main_model = update_dec_main_model(dec_main_model, decrypted_model_dict, number_of_samples)
    main_model = copy_dec_main_model_to_main_model(main_model, dec_main_model)


    test_loss, test_accuracy, recall_all, precision_all, f1_score_all, TN_all, FP_all, FN_all, TP_all, len_test_loader = validation(
        main_model, test_dl, main_criterion)


    end_time = time.time()
    test_latency = end_time - start_time



    total_comm_mb = (total_upload_all_rounds + total_download_all_rounds) / (1024 * 1024)
    local_total = local_end - local_start
    ENC_total = ENC_end - ENC_start
    sever_total = sever_end - sever_start
    DEC_total = DEC_end - DEC_start

    privacy_metrics = compute_privacy_metrics(
        model=main_model,
        member_loader=member_dl,
        nonmember_loader=nonmember_dl,
        random_state=42
    )

    mi_mean = privacy_metrics["mi_mean"]
    mi_max = privacy_metrics["mi_max"]
    kl_sym = privacy_metrics["kl_sym"]
    kl_member_nonmember = privacy_metrics["kl_member_nonmember"]
    kl_nonmember_member = privacy_metrics["kl_nonmember_member"]


    if Export:
        with open(ExportToFile, 'a', newline='\n') as out:
            writer = csv.writer(out, delimiter=',')
            if not Flag:
                header = ['Iteration', 'test accuracy', 'test Recall', 'test precision',
                          'test f1_score_all', 'test_loss', 'test latency (s)',
                          'Upload_MB_round', 'Download_MB_round', 'Total_Comm_MB_cumulative',
                          'TN_all', 'FP_all', 'FN_all', 'TP_all', 'len_test_loader',
                          'local_total', 'ENC_total', 'sever_total', 'DEC_total',
                          'MI_mean', 'MI_max', 'KL_sym', 'KL_member_nonmember', 'KL_nonmember_member',

                          ]
                writer.writerow(header)
            row = [round_idx + 1, test_accuracy, recall_all, precision_all, f1_score_all,
                   test_loss, test_latency, upload_mb_round, download_mb_round, total_comm_mb,
                   TN_all, FP_all, FN_all, TP_all, len_test_loader,
                   local_total, ENC_total, sever_total, DEC_total,
                   mi_mean, mi_max, kl_sym, kl_member_nonmember, kl_nonmember_member

                   ]
            writer.writerow(row)
        out.close()
    Flag = True
