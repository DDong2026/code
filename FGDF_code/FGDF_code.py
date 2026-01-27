
import warnings
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

pd.options.display.float_format = "{:,.4f}".format
dataset_name = 'CAN data.csv'
dataset_1 = pd.read_csv(dataset_name, header=None)
df = pd.DataFrame(dataset_1)
df = df.sample(frac=1)
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

y_train_total = 0
y_valid_total = 0
y_test_total = 0
total = 0
for i in range(2):
    y_train_total += sum(y_train == i)
    y_valid_total += sum(y_valid == i)
    y_test_total += sum(y_test == i)
    total += sum(y_train == i) + sum(y_valid == i) + sum(y_test == i)
print("=== 样本分布统计 ===")
for data_name, y_data in [("训练集", y_train), ("验证集", y_valid), ("测试集", y_test)]:
    class0 = sum(y_data == 0)
    class1 = sum(y_data == 1)
    print(f"{data_name}：正常样本={class0}（{class0/(class0+class1)*100:.1f}%），攻击样本={class1}（{class1/(class0+class1)*100:.1f}%）")

# -------------------------- 图数据转换函数 --------------------------

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
def build_sparse_edges(n_nodes, k=3):
    edges = []
    for i in range(n_nodes):
        neighbors = [(j, abs(i-j)) for j in range(n_nodes) if i != j]
        neighbors.sort(key=lambda x: x[1])
        for j, _ in neighbors[:k]:
            edges.append([i, j])
            edges.append([j, i])
    return torch.tensor(edges, dtype=torch.long).T
# -------------------------- 客户端数据划分函数 --------------------------
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


# -------------------------- 适配GAT的数据分配函数 --------------------------
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



# -------------------------- GAT模型定义 --------------------------
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


# -------------------------- GAT的加密/解密参数容器 --------------------------
class enc_model_weight:
    def __init__(self, in_channels, hidden_channels, out_channels):
        self.conv1_enc = torch.zeros(size=[hidden_channels, in_channels])
        self.conv2_enc = torch.zeros(size=[hidden_channels, hidden_channels])
        self.conv3_enc = torch.zeros(size=[hidden_channels, hidden_channels])
        self.fc_enc = torch.zeros(size=[out_channels, hidden_channels])
        self.conv1_enc_B = torch.zeros(size=[hidden_channels])
        self.conv2_enc_B = torch.zeros(size=[hidden_channels])
        self.conv3_enc_B = torch.zeros(size=[hidden_channels])
        self.fc_enc_B = torch.zeros(size=[out_channels])


class dec_model_weight:
    def __init__(self, in_channels, hidden_channels, out_channels):
        self.conv1_dec = torch.zeros(size=[hidden_channels, in_channels])
        self.conv2_dec = torch.zeros(size=[hidden_channels, hidden_channels])
        self.conv3_dec = torch.zeros(size=[hidden_channels, hidden_channels])
        self.fc_dec = torch.zeros(size=[out_channels, hidden_channels])

        self.conv1_dec_B = torch.zeros(size=[hidden_channels])
        self.conv2_dec_B = torch.zeros(size=[hidden_channels])
        self.conv3_dec_B = torch.zeros(size=[hidden_channels])
        self.fc_dec_B = torch.zeros(size=[out_channels])


# -------------------------- 客户端本地训练函数 --------------------------
def train(model, train_loader, criterion, optimizer):
    model.train()
    train_loss = 0.0
    correct = 0
    latency_list = []

    for data in train_loader:
        output = model(data)
        loss = criterion(output, data.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        start_time = time.time()
        prediction = output.argmax(dim=1, keepdim=True)
        end_time = time.time()
        latency = end_time - start_time
        latency_list.append(latency)
        correct += prediction.eq(data.y.view_as(prediction)).sum().item()

    train_loss /= len(train_loader)
    correct /= len(train_loader.dataset)
    latency_avg = sum(latency_list) / len(latency_list)

    return train_loss, correct, latency_avg


# -------------------------- 模型验证函数-------------------------
def validation(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    recall_all = 0
    precision_all = 0
    f1_score_all = 0
    TN_all = 0
    FP_all = 0
    FN_all = 0
    TP_all = 0

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


# --------------------------客户端模型初始化 --------------------------
def create_model_optimizer_criterion_dict(number_of_samples, in_channels, hidden_channels, out_channels):
    model_dict = dict()
    optimizer_dict = dict()
    criterion_dict = dict()
    encrypted_model_dict = dict()
    decrypted_model_dict = dict()

    for i in range(number_of_samples):
        model_name = "model" + str(i)
        model_info = GATNet(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels)
        model_dict.update({model_name: model_info})
        enc_m_name = "enc_model" + str(i)
        enc_m_info = enc_model_weight(in_channels, hidden_channels, out_channels)
        encrypted_model_dict.update({enc_m_name: enc_m_info})
        dec_m_name = "dec_model" + str(i)
        dec_m_info = dec_model_weight(in_channels, hidden_channels, out_channels)
        decrypted_model_dict.update({dec_m_name: dec_m_info})
        optimizer_name = "optimizer" + str(i)
        optimizer_info = torch.optim.SGD(model_info.parameters(), lr=learning_rate, momentum=momentum)
        optimizer_dict.update({optimizer_name: optimizer_info})
        criterion_name = "criterion" + str(i)
        criterion_info = nn.CrossEntropyLoss()
        criterion_dict.update({criterion_name: criterion_info})

    return model_dict, optimizer_dict, criterion_dict, encrypted_model_dict, decrypted_model_dict



def decrypt(enc):
    return enc.decrypt().tolist()


def context():
    context = ts.context(ts.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.global_scale = pow(2, 40)
    context.generate_galois_keys()
    return context


context = context()


def apply_FHE(v1):
    v1_D1 = v1.size(dim=0)
    v1_D2 = v1.size(dim=1)
    plain1 = ts.plain_tensor(v1, [v1_D1, v1_D2])
    encrypted_tensor = ts.ckks_tensor(context, plain1)
    return encrypted_tensor


# -------------------------- 加密GAT模型参数 --------------------------
def enc_model(model_dict, encrypted_model_dict, number_of_samples, epsilon, delta, sensitivity):
    with torch.no_grad():
        for i in range(number_of_samples):
            print(f"正在加密客户端 {i} 的参数...")
            model_conv1 = model_dict[name_of_models[i]].conv1.lin.weight.data.clone()
            noisy_c1 = add_gaussian_noise(model_conv1, epsilon / 2, delta, sensitivity)
            E_W1 = apply_FHE(noisy_c1)
            encrypted_model_dict[name_of_enc_models[i]].conv1_enc = E_W1
            encrypted_model_dict[name_of_enc_models[i]].conv1_enc_B = model_dict[
                name_of_models[i]].conv1.bias.data.clone()

            model_conv2 = model_dict[name_of_models[i]].conv2.lin.weight.data.clone()
            noisy_c2 = add_gaussian_noise(model_conv2, epsilon / 2, delta, sensitivity)
            E_W2 = apply_FHE(noisy_c2)
            encrypted_model_dict[name_of_enc_models[i]].conv2_enc = E_W2
            encrypted_model_dict[name_of_enc_models[i]].conv2_enc_B = model_dict[
                name_of_models[i]].conv2.bias.data.clone()

            model_conv3 = model_dict[name_of_models[i]].conv3.lin.weight.data.clone()
            noisy_c4 = add_gaussian_noise(model_conv3, epsilon / 2, delta, sensitivity)
            E_W4 = apply_FHE(noisy_c4)
            encrypted_model_dict[name_of_enc_models[i]].conv3_enc = E_W4
            encrypted_model_dict[name_of_enc_models[i]].conv3_enc_B = model_dict[
                name_of_models[i]].conv3.bias.data.clone()


            model_fc = model_dict[name_of_models[i]].fc.weight.data.clone()
            noisy_c3 = add_gaussian_noise(model_fc, epsilon / 2, delta, sensitivity)
            E_W3 = apply_FHE(noisy_c3)
            encrypted_model_dict[name_of_enc_models[i]].fc_enc = E_W3
            encrypted_model_dict[name_of_enc_models[i]].fc_enc_B = model_dict[
                name_of_models[i]].fc.bias.data.clone()

    return encrypted_model_dict


# -------------------------- 解密GAT模型参数 --------------------------
def dec_model(encrypted_model_dict, decrypted_model_dict, number_of_samples):
    with torch.no_grad():
        for i in range(number_of_samples):

            conv1_dec_w = decrypt(encrypted_model_dict[name_of_enc_models[i]].conv1_enc)
            conv2_dec_w = decrypt(encrypted_model_dict[name_of_enc_models[i]].conv2_enc)
            conv3_dec_w = decrypt(encrypted_model_dict[name_of_enc_models[i]].conv3_enc)
            fc_dec_w = decrypt(encrypted_model_dict[name_of_enc_models[i]].fc_enc)


            decrypted_model_dict[name_of_dec_models[i]].conv1_dec = conv1_dec_w
            decrypted_model_dict[name_of_dec_models[i]].conv2_dec = conv2_dec_w
            decrypted_model_dict[name_of_dec_models[i]].conv3_dec = conv3_dec_w
            decrypted_model_dict[name_of_dec_models[i]].fc_dec = fc_dec_w


            decrypted_model_dict[name_of_dec_models[i]].conv1_dec_B = encrypted_model_dict[
                name_of_enc_models[i]].conv1_enc_B
            decrypted_model_dict[name_of_dec_models[i]].conv2_dec_B = encrypted_model_dict[
                name_of_enc_models[i]].conv2_enc_B
            decrypted_model_dict[name_of_dec_models[i]].conv3_dec_B = encrypted_model_dict[
                name_of_enc_models[i]].conv3_enc_B
            decrypted_model_dict[name_of_dec_models[i]].fc_dec_B = encrypted_model_dict[
                name_of_enc_models[i]].fc_enc_B

    return decrypted_model_dict


# -------------------------- 服务器聚合GAT参数 --------------------------
def Server_get_averaged_weights(matrix_dict, number_of_samples, epsilon, delta, sensitivity, context):

    conv1_shape = encrypted_model_dict[name_of_enc_models[0]].conv1_enc.shape
    conv2_shape = encrypted_model_dict[name_of_enc_models[0]].conv2_enc.shape
    conv3_shape = encrypted_model_dict[name_of_enc_models[0]].conv3_enc.shape
    fc_shape = encrypted_model_dict[name_of_enc_models[0]].fc_enc.shape



    conv1_mean_weight = torch.zeros(size=encrypted_model_dict[name_of_enc_models[0]].conv1_enc.shape)
    conv1_mean_bias = torch.zeros(size=encrypted_model_dict[name_of_enc_models[0]].conv1_enc_B.shape)

    conv2_mean_weight = torch.zeros(size=encrypted_model_dict[name_of_enc_models[0]].conv2_enc.shape)
    conv2_mean_bias = torch.zeros(size=encrypted_model_dict[name_of_enc_models[0]].conv2_enc_B.shape)

    conv3_mean_weight = torch.zeros(size=encrypted_model_dict[name_of_enc_models[0]].conv3_enc.shape)
    conv3_mean_bias = torch.zeros(size=encrypted_model_dict[name_of_enc_models[0]].conv3_enc_B.shape)


    fc_mean_weight = torch.zeros(size=encrypted_model_dict[name_of_enc_models[0]].fc_enc.shape)
    fc_mean_bias = torch.zeros(size=encrypted_model_dict[name_of_enc_models[0]].fc_enc_B.shape)

    total_size_Enc_W_B = 0

    with torch.no_grad():
        for i in range(number_of_samples):

            conv1_mean_weight += matrix_dict['user_' + str(i)].conv1_enc
            total_size_Enc_W_B += sys.getsizeof(matrix_dict['user_' + str(i)].conv1_enc)

            conv2_mean_weight += matrix_dict['user_' + str(i)].conv2_enc
            total_size_Enc_W_B += sys.getsizeof(matrix_dict['user_' + str(i)].conv2_enc)

            conv3_mean_weight += matrix_dict['user_' + str(i)].conv3_enc
            total_size_Enc_W_B += sys.getsizeof(matrix_dict['user_' + str(i)].conv3_enc)


            fc_mean_weight += matrix_dict['user_' + str(i)].fc_enc
            total_size_Enc_W_B += sys.getsizeof(matrix_dict['user_' + str(i)].fc_enc)


            conv1_mean_bias += matrix_dict['user_' + str(i)].conv1_enc_B
            total_size_Enc_W_B += sys.getsizeof(matrix_dict['user_' + str(i)].conv1_enc_B)

            conv2_mean_bias += matrix_dict['user_' + str(i)].conv2_enc_B
            total_size_Enc_W_B += sys.getsizeof(matrix_dict['user_' + str(i)].conv2_enc_B)

            conv3_mean_bias += matrix_dict['user_' + str(i)].conv3_enc_B
            total_size_Enc_W_B += sys.getsizeof(matrix_dict['user_' + str(i)].conv3_enc_B)

            fc_mean_bias += matrix_dict['user_' + str(i)].fc_enc_B
            total_size_Enc_W_B += sys.getsizeof(matrix_dict['user_' + str(i)].fc_enc_B)


        plain_number_of_samples = ts.plain_tensor(1 / number_of_samples)
        conv1_mean_weight = conv1_mean_weight * plain_number_of_samples
        conv1_mean_bias = conv1_mean_bias / number_of_samples

        conv2_mean_weight = conv2_mean_weight * plain_number_of_samples
        conv2_mean_bias = conv2_mean_bias / number_of_samples

        conv3_mean_weight = conv3_mean_weight * plain_number_of_samples
        conv3_mean_bias = conv3_mean_bias / number_of_samples

        fc_mean_weight = fc_mean_weight * plain_number_of_samples
        fc_mean_bias = fc_mean_bias / number_of_samples


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





    return conv1_mean_weight, conv1_mean_bias, conv2_mean_weight, conv2_mean_bias, conv3_mean_weight, conv3_mean_bias, fc_mean_weight, fc_mean_bias, total_size_Enc_W_B


# -------------------------- 更新全局解密GAT模型 --------------------------
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


# -------------------------- 复制解密参数到GAT模型 --------------------------
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


# -------------------------- 更新全局加密GAT模型 --------------------------
def set_averaged_Enc_weights_as_main_Enc_model_weights_and_update_main_Enc_model(enc_main_model, conv1_mean_weight,
                                                                                 conv2_mean_weight, conv3_mean_weight,fc_mean_weight,
                                                                                 conv1_mean_bias, conv2_mean_bias,conv3_mean_bias,
                                                                                 fc_mean_bias):
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



# -------------------------- 下发全局GAT模型到客户端 --------------------------
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


# -------------------------- 下发全局加密GAT模型到客户端 --------------------------
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


# -------------------------- 收集GAT加密参数到聚合矩阵 --------------------------
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


# -------------------------- 客户端本地训练 --------------------------
def start_train_end_node_process_without_print(number_of_samples, model_dict):
    for i in range(number_of_samples):

        train_graphs = train_graph_dict[name_of_x_train_sets[i]]
        test_graphs = test_graph_dict[name_of_x_test_sets[i]]

        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_graphs, batch_size=batch_size * 2)

        model = model_dict[name_of_models[i]]
        criterion = criterion_dict[name_of_criterions[i]]
        optimizer = optimizer_dict[name_of_optimizers[i]]


        for epoch in range(numEpoch):
            train_loss, train_accuracy, train_latency = train(model, train_loader, criterion, optimizer)
            test_loss, test_accuracy, recall_all, precision_all, f1_score_all, TN_all, FP_all, FN_all, TP_all, len_test_loader = validation(
                model, test_loader, criterion)
        print(f"客户端{i} 本地训练完成：训练损失={train_loss:.4f}，训练准确率={train_accuracy:.4f}")
        print(f"客户端{i} 本地训练完成：测试损失={test_loss:.4f}，测试准确率={test_accuracy:.4f}")
        print(f"客户端{i} 本地训练完成：召回率={recall_all:.4f}，精确率={precision_all:.4f}")
        print(f"客户端{i} TN_all={TN_all:.4f}，FP_all={FP_all:.4f}，FN_all={FN_all:.4f}， TP_all={TP_all:.4f}")
    return model_dict


# -------------------------- 创建GAT聚合矩阵 --------------------------
def create_matrix_for_users(number_of_samples, in_channels, hidden_channels, out_channels):
    matrix_dict_ = {}
    for i in range(number_of_samples):
        matrix_name_user = "user_" + str(i)

        matrix_info_user = enc_model_weight(in_channels, hidden_channels, out_channels)
        matrix_dict_.update({matrix_name_user: matrix_info_user})
    return matrix_dict_


########################

batch_size = 32
learning_rate = 0.01
numEpoch = 30
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
sample_dict_train = get_subsamples(label_dict=label_dict_train, number_of_samples=number_of_samples,
                                   amount=train_amount)
train_graph_dict, y_train_dict = create_graph_subsamples(sample_dict=sample_dict_train, x_data=x_train, y_data=y_train,
                                                         x_name="x_train", y_name="y_train")


label_dict_valid = dividing_and_shuffling_labels(y_label=y_valid, seed=1, amount=valid_amount)
sample_dict_valid = get_subsamples(label_dict=label_dict_valid, number_of_samples=number_of_samples, amount=valid_amount)
valid_graph_dict, y_valid_dict = create_graph_subsamples(sample_dict=sample_dict_valid, x_data=x_valid, y_data=y_valid,
                                                         x_name="x_valid", y_name="y_valid")


label_dict_test = dividing_and_shuffling_labels(y_label=y_test, seed=1, amount=test_amount)
sample_dict_test = get_subsamples(label_dict=label_dict_test, number_of_samples=number_of_samples, amount=test_amount)
test_graph_dict, y_test_dict = create_graph_subsamples(sample_dict=sample_dict_test, x_data=x_test, y_data=y_test,
                                                       x_name="x_test", y_name="y_test")

# -------------------------- 初始化全局GAT模型 --------------------------
main_model = GATNet(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels)


print("GATConv 1 结构:", main_model.conv1)
print("GATConv 1 权重:", main_model.conv1.lin.weight.shape if hasattr(main_model.conv1, 'lin') else "No lin attribute")
print("GATConv 1 偏置:", main_model.conv1.bias.shape if main_model.conv1.bias is not None else "None")


main_optimizer = torch.optim.SGD(main_model.parameters(), lr=learning_rate, momentum=momentum)
main_criterion = nn.CrossEntropyLoss()


enc_main_model = enc_model_weight(in_channels, hidden_channels, out_channels)
dec_main_model = dec_model_weight(in_channels, hidden_channels, out_channels)


model_dict, optimizer_dict, criterion_dict, encrypted_model_dict, decrypted_model_dict = create_model_optimizer_criterion_dict(
    number_of_samples=number_of_samples,
    in_channels=in_channels,
    hidden_channels=hidden_channels,
    out_channels=out_channels
)


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
global_test_graphs = create_graph_data(x_test, y_test)
test_dl = DataLoader(global_test_graphs, batch_size=batch_size * 2)
epsilon = 1
delta = 1e-5
sensitivity = 0.1
def add_gaussian_noise(param, epsilon, delta, sensitivity):
    sigma = torch.sqrt(torch.tensor(2 * torch.log(torch.tensor(1.25 / delta))) * (sensitivity ** 2) / (epsilon ** 2))
    noise = torch.normal(mean=0.0, std=sigma.item(), size=param.shape)
    return param + noise
def generate_encrypted_gaussian_noise(shape, epsilon, delta, sensitivity, context):
    sigma = torch.sqrt(torch.tensor(2 * torch.log(torch.tensor(1.25 / delta))) * (sensitivity ** 2) / (epsilon ** 2))
    noise = torch.normal(mean=0.0, std=sigma.item(), size=shape)
    plain_noise = ts.plain_tensor(noise.numpy(), shape)
    encrypted_noise = ts.ckks_tensor(context, plain_noise)
    return encrypted_noise
def get_gaussian_sigma(epsilon, delta, sensitivity):
    return torch.sqrt(torch.tensor(2 * torch.log(torch.tensor(1.25 / delta))) * (sensitivity ** 2) / (epsilon ** 2))
#################################### Process  ####################################
ExportToFile = 'FGDF' + dataset_name
Export = True
Flag = False

for i in range(numEpoch):
    start_time = time.time()
    print(f"开始运行第 {i + 1} 轮迭代（共 {numEpoch} 轮）")
    model_dict = send_main_model_to_nodes_and_update_model_dict_before_encryption(main_model, model_dict,
                                                                                  number_of_samples)
    print(f"轮次 {i + 1}: 开始客户端训练")
    model_dict = start_train_end_node_process_without_print(number_of_samples, model_dict)
    print(f"轮次 {i + 1}: 客户端训练结束，开始加密参数")
    encrypted_model_dict = enc_model(
        model_dict,
        encrypted_model_dict,
        number_of_samples,
        epsilon,
        delta,
        sensitivity
    )
    print(f"轮次 {i + 1}: 参数加密结束，开始聚合")
    matrix_dict_ = create_matrix_for_users(
        number_of_samples=number_of_samples,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels
    )
    matrix_dict_ = send_Enc_model_to_Batching_Matrix_updates(matrix_dict_, encrypted_model_dict, number_of_samples)
    conv1_mean_weight, conv1_mean_bias, conv2_mean_weight, conv2_mean_bias, conv3_mean_weight, conv3_mean_bias,fc_mean_weight, fc_mean_bias, total_size_Enc_W_B = Server_get_averaged_weights(
        matrix_dict_,
        number_of_samples,
        epsilon,
        delta,
        sensitivity,
        context)

    conv1_mean_weight_size = sys.getsizeof(conv1_mean_weight)
    conv2_mean_weight_size = sys.getsizeof(conv2_mean_weight)
    conv3_mean_weight_size = sys.getsizeof(conv3_mean_weight)
    fc_mean_weight_size = sys.getsizeof(fc_mean_weight)
    conv1_mean_bias_size = sys.getsizeof(conv1_mean_bias)
    conv2_mean_bias_size = sys.getsizeof(conv2_mean_bias)
    conv3_mean_bias_size = sys.getsizeof(conv3_mean_bias)
    fc_mean_bias_size = sys.getsizeof(fc_mean_bias)
    total_size_global_model = conv1_mean_weight_size + conv2_mean_weight_size+ conv3_mean_weight_size + fc_mean_weight_size + conv1_mean_bias_size + conv2_mean_bias_size + conv3_mean_bias_size+ fc_mean_bias_size

    enc_main_model = set_averaged_Enc_weights_as_main_Enc_model_weights_and_update_main_Enc_model(
        enc_main_model, conv1_mean_weight, conv2_mean_weight,  conv3_mean_weight,fc_mean_weight,
        conv1_mean_bias, conv2_mean_bias, conv3_mean_bias, fc_mean_bias
    )

    encrypted_model_dict = send_Enc_model_to_nodes_and_update_Enc_model_dict(enc_main_model, encrypted_model_dict,
                                                                             number_of_samples)

    decrypted_model_dict = dec_model(encrypted_model_dict, decrypted_model_dict, number_of_samples)

    dec_main_model = update_dec_main_model(dec_main_model, decrypted_model_dict, number_of_samples)
    main_model = copy_dec_main_model_to_main_model(main_model, dec_main_model)

    test_loss, test_accuracy, recall_all, precision_all, f1_score_all, TN_all, FP_all, FN_all, TP_all, len_test_loader = validation(
        main_model, test_dl, main_criterion)

    print(f"客户端{i} 本地训练完成：测试损失={test_loss:.4f}，测试准确率={test_accuracy:.4f}")
    print(f"客户端{i} 本地训练完成：召回率={recall_all:.4f}，精确率={precision_all:.4f}")
    print(f"客户端{i} TN_all={TN_all:.4f}，FP_all={FP_all:.4f}，FN_all={FN_all:.4f}， TP_all={TP_all:.4f}")

    end_time = time.time()
    test_latency = end_time - start_time
    comm_overhead_B = total_size_Enc_W_B * 1
    comm_overhead_MB = comm_overhead_B / (1024 * 1024)
    comm_overhead_All_B = (comm_overhead_B + total_size_global_model * number_of_samples) * numEpoch
    comm_overhead = comm_overhead_All_B / (1024 * 1024)

    if Export:
        with open(ExportToFile, 'a', newline='\n') as out:
            writer = csv.writer(out, delimiter=',')
            if not Flag:
                header = ['Iteration', 'test accuracy', 'test Recall', 'test precision',
                          'test f1_score_all', 'test_loss', 'test latency', 'Communication_overhead',
                          'TN_all', 'FP_all', 'FN_all', 'TP_all', 'len_test_loader']
                writer.writerow(header)
            row = [i + 1, test_accuracy, recall_all, precision_all, f1_score_all,
                   test_loss, test_latency, comm_overhead, TN_all, FP_all, FN_all, TP_all, len_test_loader]
            writer.writerow(row)
        out.close()
    Flag = True

print("联邦学习GCN模型训练完成！结果已保存到", ExportToFile)