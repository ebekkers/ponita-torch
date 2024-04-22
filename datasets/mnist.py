import os
import torch
import requests
import zipfile
from torch.utils.data import Dataset, DataLoader


class MNISTSuperpixels(Dataset):
    url = 'https://data.pyg.org/datasets/MNISTSuperpixels.zip'
    file_name = 'MNISTSuperpixels.zip'

    def __init__(self, root, train=True):
        self.root = root
        self.train = train
        self.raw_data_path = os.path.join(self.root, 'MNISTSuperpixels.pt')
        self.data_file = 'train_data.pt' if self.train else 'test_data.pt'
        self.data_path = os.path.join(self.root, self.data_file)
        
        if not os.path.exists(self.data_path):
            self.download_and_process()

        self.data = torch.load(self.data_path)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def download_and_process(self):
        # Download
        download_path = os.path.join(self.root, self.file_name)
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        response = requests.get(self.url)
        with open(download_path, 'wb') as f:
            f.write(response.content)

        # Extract
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(self.root)
        os.remove(download_path)

        data = torch.load(self.raw_data_path)
        torch.save(data[0], os.path.join(self.root, 'train_data.pt'))
        torch.save(data[1], os.path.join(self.root, 'test_data.pt'))
        os.remove(self.raw_data_path)


def collate_fn(batch):
    pos, x, y, batch_idx, edge_index_batch = [], [], [], [], []
    cum_nodes = 0
    for i, item in enumerate(batch):
        num_nodes = item['x'].shape[0]
        pos.append(item['pos'])
        x.append(item['x'])
        y.append(item['y'])
        batch_idx.extend([i] * num_nodes)
        edge_index = item['edge_index'] + cum_nodes  # Offset node indices
        edge_index_batch.append(edge_index)
        cum_nodes += num_nodes
    pos = torch.cat(pos, dim=0)
    x = torch.cat(x, dim=0)
    y = torch.stack(y, dim=0)
    batch_idx = torch.tensor(batch_idx, dtype=torch.long)
    edge_index = torch.cat(edge_index_batch, dim=1)

    return {'pos': pos, 'x': x, 'y': y.squeeze(), 'batch': batch_idx, 'edge_index': edge_index}

# # Example usage:
# root_dir = './data'
# dataset = MNISTSuperpixelsDataset(root=root_dir, train=True)
# dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# for data, labels in dataloader:
#     print(data.shape, labels.shape)
#     break
