import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

torch.set_num_threads(5)
device = torch.device('cuda:0')
device_id =0
batch_size = 256
path = r'../1D data/1Ddata_200000.pkl'
# Read the CSV file into a DataFrame
df = pd.read_pickle(path)



def normalized_data(data):
    data = torch.FloatTensor(data)
    data[data<1e-8]=1e-8
    data = torch.log10(data)
    data /=8
    return data

#make dataloader
train_df = df[df['assignment'] == 'training']
valid_df = df[df['assignment'] == 'validation']
test_df = df[df['assignment'] == 'test']


input_pz = normalized_data(train_df['p0_z']).clone()
output_soln = torch.FloatTensor(train_df['Stored_Soln'])
data_min = torch.min(output_soln, dim=1, keepdim=True).values
data_max = torch.max(output_soln, dim=1, keepdim=True).values


output_soln = (output_soln - data_min) / (data_max - data_min)
out = output_soln[:,0:36]


from annoy import AnnoyIndex

DIMENSIONS = 36
def build_annoy_index(data, n_trees=9):
    t = AnnoyIndex(DIMENSIONS, 'euclidean')  
    for i, item in enumerate(data):
        t.add_item(i, item)
    t.build(n_trees)  
    return t

def find_similar_pairs(index, data, n_neighbors=1):
    similar_pairs = []
    
    for i, item in enumerate(data):
        neighbors = index.get_nns_by_item(i, n_neighbors)
        
        # Exclude the item itself
        for neighbor in neighbors[1:]:
            similar_pairs.append((i, neighbor))
            
    return similar_pairs


index = build_annoy_index(out)
pairs = find_similar_pairs(index, out)


def get_delete_indices(pairs):
    occurrence_dict = {}
    for i, j in pairs:
        if i not in occurrence_dict:
            occurrence_dict[i] = {'seen': True, 'has_similar': True}
        if j not in occurrence_dict:
            occurrence_dict[j] = {'seen': False, 'has_similar': True}
    delete_indices = [index for index, value in occurrence_dict.items() if not (value['seen'] or not value['has_similar'])]
    return delete_indices

indices_to_delete = get_delete_indices(pairs)
mask = torch.ones(out.size(0), dtype=torch.bool)
mask[indices_to_delete] = False
filtered_output = out[mask]
filtered_input = input_pz[mask]
print('Done')
print(filtered_output.shape)
print(filtered_input.shape)


class CDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        input_pz = normalized_data(self.dataframe.iloc[idx]['p0_z']).clone()
        output_soln = torch.FloatTensor(self.dataframe.iloc[idx]['Stored_Soln']).clone()
        output_soln = output_soln[0:36]
        output_soln = (output_soln - output_soln.min()) / (output_soln.max() - output_soln.min())
        return input_pz, output_soln
    
class input_set(Dataset):
    def __init__(self, input, output):
        self.input = input
        self.output = output

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx):
        return self.input[idx], self.output[idx]

train_dataset = input_set(filtered_input, filtered_output)
validate_dataset = CDataset(valid_df)
test_dataset = CDataset(test_df)
print(len(train_dataset))
print(len(validate_dataset))
print(len(test_dataset))

training_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)




# In[8]:


from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom


ndim_tot = 36
def subnet_fc(c_in, c_out):
    return nn.Sequential(nn.Linear(c_in, 512), nn.ReLU(), nn.Linear(512, 1024), nn.ReLU(),nn.Linear(1024, 512), nn.ReLU(),
                         nn.Linear(512,  c_out))

nodes = [InputNode(ndim_tot, name='input')]

for k in range(15):
    nodes.append(Node(nodes[-1],
                      GLOWCouplingBlock,
                      {'subnet_constructor':subnet_fc, 'clamp':1},
                      name=F'coupling_{k}'))
    nodes.append(Node(nodes[-1],
                      PermuteRandom,
                      {'seed':k},
                      name=F'permute_{k}'))

nodes.append(OutputNode(nodes[-1], name='output'))
INN_model = ReversibleGraphNet(nodes, verbose=False).to(device)

class MyModel(nn.Module):
    def __init__(self, INN):
        super(MyModel, self).__init__()
        self.reversible_part = INN
        
    def forward(self, x):
        x, log_jac_det  = self.reversible_part(x)
        return x, log_jac_det


    def invert_model(self, final_output):
        # Invert through linear layer
        # Now, invert through the reversible part
        x_reverse = self.reversible_part(final_output, rev = True)[0]
        return x_reverse


def MMD_multiscale(x, y):
    xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2.*xx
    dyy = ry.t() + ry - 2.*yy
    dxy = rx.t() + ry - 2.*zz

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    for a in [0.2, 1.5, 3]:
        XX += a**2 * (a**2 + dxx)**-1
        YY += a**2 * (a**2 + dyy)**-1
        XY += a**2 * (a**2 + dxy)**-1

    return torch.mean(XX + YY - 2.*XY)



def seed_everything(seed):
    """
    Changes the seed for reproducibility. 
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  
    

model_name = '1D_modeINN_new.pth'
def train_model(INNmodel):
    zeros_noise_scale = 5e-4
    resume = 0
    model = MyModel(INNmodel)
    model = model.to(device)
    if resume == 1:
        save_mse = 0.04
        model.load_state_dict(torch.load(model_name, map_location=device))
    else:
        save_mse = 100
    epoch = 300
    #criterion = nn.MSELoss()
    criterion = nn.HuberLoss()
    l_mmd = MMD_multiscale
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay= 1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
    train_loss = []
    valid_loss =[]
    
    for epoches in range(epoch):
        with tqdm(training_dataloader, unit="batch") as tepoch:
            model.train()
            running_loss = 0.0

            for train_input, train_target in tepoch:
                train_input = train_input.float().to(device)
                dim_x = train_input.shape[1]
                train_target = train_target.float().to(device)
                
                pad_input = zeros_noise_scale * torch.randn(train_input.shape[0], 36-11).to(device)
                x = torch.cat((train_input, pad_input), dim =1)
                
                y = train_target
                
                optimizer.zero_grad()
                z, log_jac_det = model(y)  #z = [x, z]
                encode_x = z[:,0:dim_x]
                encode_noise = z[:,dim_x:]
                
                
                loss_z = torch.mean(0.5 * encode_noise.pow(2).sum(dim=1) - log_jac_det)
                
                output_block_grad = torch.cat((z[:,:dim_x].detach(), z[:,dim_x:]),dim=1)
       
                loss_reconstrut = criterion(encode_x, train_input)
        
                loss_independent = l_mmd(output_block_grad, x)
                
                l = 5*loss_reconstrut  + 0.001*loss_z  +  loss_independent           
                test_sample_output = F.pad(train_input, (0, 25), 'constant', 0)
                INN_predict_output = model.invert_model(test_sample_output.to(device))
                loss_invert = criterion(INN_predict_output, train_target)
                loss_invert = l +5*loss_invert 
                loss_invert.backward()
                
                running_loss += l.item()+ loss_invert.item()
                optimizer.step()
                tepoch.set_postfix(ul = loss_reconstrut.item(), ind = loss_independent.item(), rev = loss_invert.item()) 
                tepoch.set_description(f"epoch %2f " % epoches)
            scheduler.step()
        print(f'Epoch {epoches+1}/{epoch}, Loss: {running_loss/len(training_dataloader):.4f}')
        run_loss = running_loss/len(training_dataloader)
        train_loss.append(run_loss)

        model.eval()
        total_loss = 0.0
        total_data_points = 0
        total_ssres = 0.0
        mean_ssres = 0
        with torch.no_grad():
            for inputs, target in tqdm(validate_dataloader):
                inputs = inputs.float().to(device)
                target = target.float().to(device)
                output, _ = model(target)
                loss = F.mse_loss(output[:, :11], inputs, reduction='sum')
                ssres = (output[:, :11] - inputs).pow(2).sum()

                total_loss += loss.item() 
                total_ssres += ssres.item()
                total_data_points += len(inputs)
        avg_mse_loss = total_loss / total_data_points
        ave_ssre_loss = total_ssres / total_data_points
        print(f"Average MSE Loss on Valid Set: {avg_mse_loss}")
        valid_loss.append(avg_mse_loss)
        if avg_mse_loss< save_mse:
            torch.save(model.state_dict(), model_name)
            save_mse = avg_mse_loss
    return train_loss, valid_loss
        


def test(test_loader, model):
    mean_loss = 0
    model.eval()    
    total_loss = 0.0
    total_ssres = 0
    total_data_points = 0
    mean_ssres = 0
    model = model.to(device)
    with torch.no_grad():
        for inputs, target in tqdm(test_loader):  
            target = target.to(device)
            output, _= model(target.float().to(device))
            inputs = inputs.to(device)
            loss = F.mse_loss(output[:, :11], inputs, reduction='sum')
            ssres = (output[:, :11] - inputs).pow(2).sum()

            total_loss += loss.item() 
            total_ssres += ssres.item()
            total_data_points += len(inputs)
    avg_mse_loss = total_loss / total_data_points
    ave_ssre_loss = total_ssres / total_data_points
    return avg_mse_loss, ave_ssre_loss
    


###add those code before your training loop
import subprocess  
import threading  
import time
import pickle

def compute_average_hourly_utilization(minute_utilizations):
    padding_length = (60 - len(minute_utilizations) % 60) % 60 
    padded_utilizations = minute_utilizations + [0] * padding_length
    num_hours = len(padded_utilizations) // 60
    hourly_averages = [sum(padded_utilizations[i*60:(i+1)*60])/60.0 for i in range(num_hours)]

    return hourly_averages

def get_gpu_utilization(device_id):
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
        ).decode("utf-8").strip().split('\n')
        return int(result[device_id])######
    except Exception as e:
        print(f"Error querying GPU utilization: {e}")
        return None

# Function to repeatedly track GPU utilization every minute
def track_gpu_utilization_periodically(device_id):
    if not stop_tracking:
        utilization = get_gpu_utilization(device_id)
        if utilization is not None:
            utilization_list.append(utilization)
        threading.Timer(60, track_gpu_utilization_periodically, args=[device_id]).start()


utilization_list = []
stop_tracking = False
track_gpu_utilization_periodically(device_id) #  Start the periodic GPU tracking
start_time = time.time()    
    
seed_everything(0)    
tl,vl = train_model(INN_model)

stop_tracking = True
end_time = time.time()
total_time = end_time - start_time

# Print GPU utilizations
print("***********************************************")
print("energy results")
print("\nGPU Utilizations recorded every minute:", utilization_list)

with open('utilization_data.pkl', 'wb') as f:
    pickle.dump(utilization_list, f)
    
average_hourly_utilizations = compute_average_hourly_utilization(utilization_list)
print("Average hourly GPU Utilizations:", average_hourly_utilizations)
print(f"\nTotal training time: {total_time:.2f} seconds.")
print("***********************************************")



print("test procedure")
model = MyModel(INN_model)
model.load_state_dict(torch.load(model_name, map_location=device))
avg_mse_loss, ave_ssre_loss = test(test_dataloader, model)
print(f"Average MSE Loss on Test Set: {avg_mse_loss}")



