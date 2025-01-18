import torch
import torch.nn as nn
import torch.nn.functional as F
from flexgen.utils import GB, T, MB, KB
from flexgen.llama_config import get_llama_config
import argparse
import torch.optim as optim
import numpy as np

import math

# c1: float = 0.0168
# c2: float = 0.0328
# c3: float = 0.0621

# def round_to_nearest_power_of_two(n):
#     # Special case for non-positive numbers
#     if n <= 0:
#         raise ValueError("Input must be a positive number.")

#     # Compute the exponent for the nearest power of two
#     exponent = round(math.log(n, 2))
    
#     # Return 2 raised to this exponent
#     return 2 ** exponent

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predicted, actual):
        mse_loss = F.mse_loss(predicted, actual, reduction='mean')
        rmse_loss = torch.sqrt(mse_loss)
        return rmse_loss

# Example data - replace these with your actual data
# bls -> effective batch size = num_of_batch x batch_size
input_data = [
    {'bls': 1, 'gbs': 1, 'wc': 1, 'hc': 0, 'cg': 1, 'cc': 0, 'wn': 0, 'hn': 0, 'n': 256, 'throughput':  0.195, 'prompt_len' : 2048},
    {'bls': 1, 'gbs': 1, 'wc': 1, 'hc': 0, 'cg': 1, 'cc': 0, 'wn': 0, 'hn': 0, 'n': 256, 'throughput':  0.195, 'prompt_len' : 2048},

    {'bls': 1, 'gbs': 1, 'wc': 0, 'hc': 0, 'cg': 0, 'cc': 0, 'wn': 1, 'hn': 0, 'n': 256, 'throughput':  0.102, 'prompt_len' : 2048},
    {'bls': 2, 'gbs': 2, 'wc': 0, 'hc': 0, 'cg': 0, 'cc': 0, 'wn': 1, 'hn': 0, 'n': 256, 'throughput':  0.216, 'prompt_len' : 2048},

]

# Latency / Time
targets = [
    1312.196,
    1313.112,

    2512.968,
    2373.940
]

class ComputationModel(nn.Module):
    def __init__(self, config):
        super(ComputationModel, self).__init__()
        # Define the variables as trainable parameters
        self.ctog_bdw = nn.Parameter(torch.tensor(18.364295959472656))
        self.gtoc_bdw_cache = nn.Parameter(torch.tensor(0.9700000286102295))
        self.gtoc_bdw_hidden = nn.Parameter(torch.tensor(3.7059247493743896))

        self.dtoc_bdw = nn.Parameter(torch.tensor(9.185635566711426))
        self.ctod_bdw_cache_p = nn.Parameter(torch.tensor(0.4802015423774719))
        self.ctod_bdw_hidden_p = nn.Parameter(torch.tensor(1.2484183311462402))
        self.ctod_bdw_g = nn.Parameter(torch.tensor(2.015000104904175))

        self.mm_flops_p = nn.Parameter(torch.tensor(21.979633331298828))
        self.mm_flops_g = nn.Parameter(torch.tensor(4.300000190734863))
        self.bmm_flops_p = nn.Parameter(torch.tensor(10.592555046081543))
        self.bmm_flops_g = nn.Parameter(torch.tensor(0.07900000363588333))
        self.cpu_flops = nn.Parameter(torch.tensor(0.012299999594688416))

        # Store the configuration parameters
        self.config = config

    def forward(self, bls, gbs, wc, hc, cg, cc, wn, hn, prompt_len, n):
        # Ensure that batch local size and global batch size are positive and that the former is a multiple of the latter
        assert bls > 0 and gbs > 0
        assert bls >= gbs and bls % gbs == 0
        s = prompt_len
        l = self.config.num_hidden_layers
        h1 = self.config.input_dim
        h2 = self.config.intermediate_size

        cn = 1 - cc - cg

        # layer weight size
        wi = 8 * h1 ** 2 + 4 * h1 * h2
        # cpu_flops_real = self.cpu_flops * np.maximum(0.1,
        #     1 + c1 * (max(0, math.log2(64 / gbs)) * max(0, math.log2(4096 / h1)))
        #     - c2 * max(0, math.log2(64 / gbs))
        #     - c3 * max(0, math.log2(4096 / h1)))

        ctogp = (1 / (self.ctog_bdw * GB)) * (wi * (wc + wn)
                     + 2 * s * h1 * bls * (hc + hn))
        
        gtocp = (1 / (self.gtoc_bdw_cache * GB)) * (4 * (s + 1) * h1 * bls * (cc + cn)) \
                   + (1 / (self.gtoc_bdw_hidden * GB)) * 2 * s * h1 * bls * (hc + hn)
        
        dtocp = (1 / (self.dtoc_bdw * GB)) * (wi * wn + 2 * s * h1 * bls * hn)

        # ctodp = (cache_ctodp + hidden_ctodp) / ctod_bdw
        ctodp = (1 / (self.ctod_bdw_cache_p * GB)) * 4 * bls * (s + 1) * h1 * cn \
                   + (1 / (self.ctod_bdw_hidden_p * GB)) * 2 * s * h1 * bls * hn

        # compp = gpu_compp
        compp = (1 / (self.mm_flops_p * T)) * bls * (8 * s * h1 ** 2  + 4 * s * h1 * h2) \
                     + (1 / (self.bmm_flops_p * T)) * 4 * bls * s ** 2 * h1

        ctogg = (1 / (self.ctog_bdw * GB)) * (wi * (wc + wn)
                     + 2 * h1 * bls * (hc + hn))

        # gtocg = hidden_gtocg / gtoc_bdw
        gtocg = (1 / (self.gtoc_bdw_hidden * GB)) * 2 * h1 * bls * (hc + hn)

        # dtocg = (cache_dtocg + weight_dtocg + hidden_dtocg) / dtoc_bdw
        dtocg = (1 / (self.dtoc_bdw * GB)) * (4 * bls * (s + n / 2) * h1 * cn
                                       + 2 * h1 * bls * hn) \
                     + (1 / (self.dtoc_bdw * GB * 0.95)) * wi * wn 

        # ctodg = (cache_ctodg + hidden_ctodg) / ctod_bdw
        ctodg = (1 / (self.ctod_bdw_g * GB)) * (4 * bls * h1 * cn
                     + 2 * h1 * bls * hn)
        compg = (1 / (self.mm_flops_g * T)) * bls * (8 * h1 ** 2  + 4 * h1 * h2) \
                     + (1 / (self.bmm_flops_g * T)) * 4 * bls * (s + n / 2) * h1 * cg \
                     + (1 / (self.cpu_flops * T)) * 4 * bls * (s + n / 2) * h1 * (cc + cn)
        return max([ctogp, gtocp, dtocp, ctodp, compp]) * l + max([ctogg, gtocg, dtocg, ctodg, compg]) * (n - 1) * l

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="huggyllama/llama-65b")
    args = parser.parse_args()

    # Create an instance of the model
    config = get_llama_config(args.model)
    model = ComputationModel(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # Optimizer
    # optimizer = optim.Adam(model.parameters(), lr= 1e-2)
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-9)
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    # Loss function
    loss_fn = RMSELoss()

    # Early stopping parameters
    patience = 1000  # Number of epochs to wait for improvement before stopping
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False

    # Training loop
    num_epochs = 2000
    for epoch in range(num_epochs):
        total_loss = 0
        for data_point, target_value in zip(input_data, targets):
            optimizer.zero_grad()
    # {'bls': 0.427 / 64, 'gbs': 1, 'ctog_bdw': 0.000107 * GB, 'wc': 0, 'hc': 0, 'gtoc_bdw_hidden': 0.000107 * GB, 'cg': 100, 'cc': 0, 'n': 64, 'throughput': 0.427, 'compress_w' : True, 'compress_cache' : True}
            # Convert data to tensors
            bls = torch.tensor(data_point['bls']).to(device)
            gbs = torch.tensor(data_point['gbs']).to(device)
            wc = torch.tensor(data_point['wc']).to(device)
            hc = torch.tensor(data_point['hc']).to(device)
            cg = torch.tensor(data_point['cg']).to(device)
            cc = torch.tensor(data_point['cc']).to(device)
            wn = torch.tensor(data_point['wn']).to(device)
            hn = torch.tensor(data_point['hn']).to(device)
            prompt_len = torch.tensor(data_point['prompt_len']).to(device)
            n = torch.tensor(data_point['n']).to(device)
            throughput = torch.tensor(data_point['throughput']).to(device)
            # Convert target to tensor
            target = torch.tensor(target_value).to(device)

            # Forward pass
            output = model(bls, gbs, wc, hc, cg, cc, wn, hn, prompt_len, n)  # Pass the tensors to the model
            # Calculate loss
            loss = loss_fn(output, target)
            total_loss += loss.item()

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Enforce non-negativity constraint using abs()
            with torch.no_grad():
                for param in model.parameters():
                    param.abs_()  # In-place absolute value
            
        # Print average loss for the epoch
        avg_loss = total_loss / len(input_data)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.2f}')
        # Check for improvement
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping condition
        if epochs_no_improve == patience:
            print("Early stopping triggered")
            early_stop = True
            break
    
    # If early stopping was triggered, load the best model state
    if early_stop:
        model.load_state_dict(best_model_state)
    
    # After the training loop
    trained_ctog_bdw = model.ctog_bdw.item()
    trained_gtoc_bdw_cache = model.gtoc_bdw_cache.item()
    trained_gtoc_bdw_hidden = model.gtoc_bdw_hidden.item()

    dtoc_bdw = model.dtoc_bdw.item()
    ctod_bdw_cache_p = model.ctod_bdw_cache_p.item()
    ctod_bdw_hidden_p = model.ctod_bdw_hidden_p.item()
    ctod_bdw_g = model.ctod_bdw_g.item()

    trained_mm_flops_p = model.mm_flops_p.item()
    trained_mm_flops_g = model.mm_flops_g.item()
    trained_bmm_flops_p = model.bmm_flops_p.item()
    trained_bmm_flops_g = model.bmm_flops_g.item()
    trained_cpu_flops = model.cpu_flops.item()

    print("Trained ctog_bdw:", trained_ctog_bdw)
    print("Trained gtoc_bdw_cache:", trained_gtoc_bdw_cache)
    print("Trained gtoc_bdw_hidden:", trained_gtoc_bdw_hidden)

    print("Trained dtoc_bdw:", dtoc_bdw)
    print("Trained ctod_bdw_cache_p:", ctod_bdw_cache_p)
    print("Trained ctod_bdw_hidden_p:", ctod_bdw_hidden_p)
    print("Trained ctod_bdw_g:", ctod_bdw_g)

    print("Trained mm_flops_p:", trained_mm_flops_p)
    print("Trained mm_flops_g:", trained_mm_flops_g)
    print("Trained bmm_flops_p:", trained_bmm_flops_p)
    print("Trained bmm_flops_g:", trained_bmm_flops_g)
    print("Trained cpu_flops:", trained_cpu_flops)


