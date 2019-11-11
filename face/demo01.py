from __future__ import print_function, division
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

class Model(nn.Module)：
    # Our model

    def __init__(self, input_size, output_size)：
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input)：
        output = self.fc(input)
        print("\tIn Model： input size", input.size(),
              "output size", output.size())

        return output


if __name__=="__main__":
    input_size = 5
    data_size = 40
    batch_size = 8
    
    model = Model(input_size, output_size)

    device = torch.device("cuda")

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    model.to(device)

    rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)
    print(len(rand_loader))

    for data in rand_loader：
    input = data.to(device)
    output = model(input)
    print("Outside： input size", input.size(),
          "output_size", output.size())

    
