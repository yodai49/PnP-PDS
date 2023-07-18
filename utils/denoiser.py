import torch
import torch.nn as nn

class Denoiser:
    def __init__(self, file_name):
        self.cost = 0
        self.network = load_network(file_name)

    def denoise(self, x):
        out = apply_model(x, self.network)
        return out
    
def load_network(file_name):
    net = simple_CNN()
    file_name = file_name
    model = nn.DataParallel(net).cuda()
    checkpoint = torch.load(file_name, map_location= lambda storage, loc: storage)
    model.module.load_state_dict(checkpoint.module.state_dict())
    return model.eval()

def apply_model(x, network):
    x_in = torch.from_numpy(x)
    x_out = network(x_in)
    return x_out

class simple_CNN(nn.Module):
    def __init__(self):
        super(simple_CNN, self).__init__()
        self.depth = 5

        n_ch_in = 3
        n_ch = 64
        n_ch_out = 3
        self.in_conv = nn.Conv2d(n_ch_in, n_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_list = nn.ModuleList([nn.Conv2d(n_ch, n_ch, kernel_size=3, stride=1, padding=1, bias=True) for _ in range(self.depth-2)])
        self.out_conv = nn.Conv2d(n_ch, n_ch_out, kernel_size=3, stride=1, padding=1, bias=True)
        self.nl_list = nn.ModuleList([nn.LeakyReLU() for _ in range(self.depth-1)])