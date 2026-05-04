"""
ims_contrastive_model/optimization.py
---------------------------------------
Helper functions to dynamically adapt network architecture 
and execute the training loops.
"""

# python
import copy
import time

# numerical
import numpy as np

# torch 
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# IMS 
import m2aia as m2
from .architecture import ContrastiveAutoencoder
from .dataloader import IMSPyTorchDataset
## functions for find peaks and their envelopes 
from scipy.signal import find_peaks, peak_widths

# ---------------------
# train loop
# ---------------------

def train_loop_ims_contrastive_model(
        model: ContrastiveAutoencoder, 
        dataloader: DataLoader, 
        criterion,  # it is forward from autoencoder 
        device, 
        epochs: int, 
        lr: float, 
        patience_limit: int,
        save_callback: callable # function
        ):
    '''
    Quick explanation of loops 
    '''
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    t_max = epochs // 10 if epochs >= 10 else 1
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max)

    best_loss = float('inf')
    patience = 0
    
    total_pixels = len(dataloader.dataset) # Total number of spectra to process

    # TODO - insert noise peaks 
    PEAK_BANK = precompute_peak_bank(dataloader.dataset, 2)

    for epoch in range(epochs):
        # set epoch
        ## set train mode
        model.train()
        ## add row values 
        epoch_acc = {
            'contrastive_loss': 0.0, 'std_loss': 0.0, 'mean_loss': 0.0, 
            'decoder_loss': 0.0, 'total_loss': 0.0
        }
        
        # progress tracking for the current epoch
        processed_pixels = 0
        last_log_percent = 0
        start_time = time.time()

        print(f"\n--- Epoch {epoch+1}/{epochs} ---")

        # go through batches
        for i, batch in enumerate(dataloader):
            # TODO add here some visualization of training 
            ## batch = (batch, intensities ~ resampled)
            x1 = batch.to(device)
            x2 = apply_noise(x1, dataloader.dataset, PEAK_BANK)
            
            optimizer.zero_grad()
            z1, x1_hat = model(x1)
            z2, x2_hat = model(x2)
            
            loss, loss_dict = criterion(
                emb_i=z1, 
                emb_j=z2, 
                encoder_inputs=torch.cat([x1, x2]), 
                decoder_outputs=torch.cat([x1_hat, x2_hat])
            )
            
            loss.backward()
            optimizer.step()
            
            for k in epoch_acc.keys():
                epoch_acc[k] += loss_dict[k]

            ### LOGS[cite: 6]

            #### Update progress tracking
            processed_pixels += len(batch)
            current_percent = (processed_pixels / total_pixels) * 100

            #### Log progress every ~5% or at the end
            if current_percent - last_log_percent >= 5 or processed_pixels == total_pixels:
                elapsed_time = time.time() - start_time
                # Calculate remaining time based on current speed
                remaining_time = (elapsed_time / processed_pixels) * (total_pixels - processed_pixels)
                
                print(f"[{current_percent:3.0f}%] {processed_pixels}/{total_pixels} pixels processed | "
                      f"Loss: {loss.item():.4f} | ETA: {remaining_time/60:.1f} min")
                last_log_percent = current_percent
        
        # update model
        scheduler.step()

        ## epoch mean results
        avg_metrics = {k: v / len(dataloader) for k, v in epoch_acc.items()}
        avg_metrics['epoch'] = epoch + 1

        ## logging 
        current_loss = avg_metrics['total_loss']
        if current_loss < best_loss:
            best_loss = current_loss
            patience = 0
            ### Run save callback for keeping iteration | overwrite best model
            save_callback(avg_metrics, is_best=True)
        else:
            patience += 1
            ### Run save callback for keeping iteration
            save_callback(avg_metrics, is_best=False)
            
        print(f"Summary Epoch {epoch+1} | Mean Loss: {current_loss:.4f} | Patience: {patience}/{patience_limit}")

        ## Early stopping 
        if patience > patience_limit:
            print(f"[Optimization] Early stopping triggered at epoch {epoch}.")
            break






# ---------------------
# helpers
# ---------------------


import numpy as np
import copy

def suggest_cnn_configuration(IMSLoader: IMSPyTorchDataset, latent_dim: int, hyperparameters: dict = None): 
    """
    Suggests CNN hyperparameters optimized for IMS data.
    If hyperparameters is provided, it returns them directly.
    Otherwise, it calculates a configuration based autor intuition.
    """
    input_dim = IMSLoader.GetGridXAxisDepth()
    
    # If hyperparameters are provided, assume 
    if hyperparameters is not None:
        params = copy.deepcopy(hyperparameters)
        params['input_dim'] = input_dim
        params['latent_dim'] = latent_dim
        return params

    # Initial size
    ## Estimate envelope width
    auto_kernel_1 = estimate_max_peak_width(IMSLoader, sample_size=10_000)
    print(f"[Optimization] Estimated peak envelope width: {auto_kernel_1} bins")

    # Layer 1: Wide kernel (15) for peak envelope detection (~0.15 Da at 0.01 Da res)
    # High channel count (256) to capture diverse chemical signatures
    channels = [1, 64, 32, 16, 8]
    kernels = [auto_kernel_1, 7, 5, 3]
    strides = [3, 4, 4, 3]

    # Dynamic reduction logic based on latent_dim
    # Goal: Ensure the flattened conv output is roughly 2x-4x the latent_dim
    current_stride_prod = np.prod(strides)
    current_out_dim = input_dim // current_stride_prod
    
    # target_conv_out is set to be proportional to latent_dim (max ~400)
    target_conv_out = max(latent_dim * 2, 512) 

    # Add layers if the dimensionality is still too high for the latent_dim bottleneck
    while current_out_dim > target_conv_out and len(channels) < 6:
        new_stride = 2
        strides.append(new_stride)
        # we cap at 8 channels 
        channels.append(max(channels[-1] // 2, 8))
        kernels.append(3)
        current_stride_prod *= new_stride
        current_out_dim = input_dim // current_stride_prod

    test_params = {
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'channels': channels,
        'kernels': kernels,
        'strides': strides
    }

    # # OLD - FROM ARTICLE FOR MOUSE BLADDER 
    # test_params = {
    #     # deterministic
    #     'input_dim': input_dim,
    #     'latent_dim': latent_dim,
    #     # predicted:  
    #     'channels': [1, 2, 4, 16, 32, 64],
    #     'kernels': [7, 7, 5, 5, 5],
    #     'strides': [2, 3, 3, 3, 3]
    # }

    return test_params


def apply_noise(vec: torch.Tensor, IMSDataset: IMSPyTorchDataset, PeakBank) -> torch.Tensor:
    """Adds biological/chemical noise to batch by injecting random peaks 
    sampled from a precomputed bank to avoid CPU-GPU overhead."""
    # local names
    dataset = IMSDataset
    # obtain vector and params 
    norm_type = dataset.img.normalization
    device = vec.device
    noisy_batch = vec.clone()
    
    # Check if bank exists, if not, the function returns original (failsafe)
    if not hasattr(dataset, 'peak_bank'):
        return noisy_batch

    # Identify peaks in the current batch to determine noise scale
    ## We use a fast GPU-based local maxima detection instead of scipy
    shifted_left = torch.cat([vec[:, 1:], vec[:, -1:]], dim=1)
    shifted_right = torch.cat([vec[:, :1], vec[:, :-1]], dim=1)
    mean_vals = vec.mean(dim=1, keepdim=True)
    
    ## Peak is higher than neighbors and higher than mean
    peak_mask = (vec > shifted_left) & (vec > shifted_right) & (vec > mean_vals)
    num_peaks_in_batch = peak_mask.sum(dim=1)
    
    # Determine number of foreign peaks to add (up to 5%, at least 1)
    num_peaks_to_add = torch.clamp((num_peaks_in_batch * 0.05).int(), min=1)
    max_iterations = num_peaks_to_add.max().item()

    # Iterate through the maximum required additions
    for p_step in range(max_iterations):
        ## Mask for samples that still need peaks added
        active_mask = (num_peaks_to_add > p_step).nonzero(as_tuple=True)[0]
        if len(active_mask) == 0:
            break
            
        ## Randomly select peaks from the precomputed bank for the active batch
        rand_bank_idxs = torch.randint(0, len(PeakBank), (len(active_mask),))
        
        for i, batch_idx in enumerate(active_mask):
            start, end, peak_envelope = PeakBank[rand_bank_idxs[i]]
            ## Add the peak envelope directly on GPU
            noisy_batch[batch_idx, start:end] += peak_envelope.to(device)

    # Re-normalize - data consistency 
    if norm_type == 'TIC':
        ### Total Ion Count normalization: ensure sum of intensities is constant
        noisy_tic = torch.sum(noisy_batch, dim=1, keepdim=True)
        noisy_batch = noisy_batch / noisy_tic.clamp(min=1e-12)
        
    return noisy_batch

# VERY SLOW
# def apply_noise(vec: torch.Tensor, IMSDataset: IMSPyTorchDataset) -> torch.Tensor:
#     """Adds biological/chemical noise to batch by injecting random peaks 
#     sampled from the entire IMS image up to 5% of peaks."""
#     # local names
#     dataset = IMSDataset
#     # obtain vector and params 
#     norm_type = dataset.img.normalization # e.g., 'TIC'
#     device = vec.device
#     noisy_batch = vec.clone()
    
#     # Iterate through batch spectra
#     for i in range(vec.shape[0]):
#         ## convert spectrum to cpu (scipy library)
#         spectrum_np = vec[i].cpu().numpy()
        
#         ## Identify peaks in the current spectrum to determine noise scale
#         ### We use scipy.signal for robust 1D peak detection
#         ### We consider only values above mean ys value 
#         peaks, _ = find_peaks(spectrum_np, height=np.mean(spectrum_np))
        
#         ## Determine number of foreign peaks to add (up to 5%, at least 1)
#         num_peaks_to_add = max(1, int(len(peaks) * 0.05))
        
#         ## Iterate through dataset to obtain new peaks
#         for _ in range(num_peaks_to_add):
#             ### Pick a random spectrum index from the full image
#             rand_idx = np.random.randint(0, len(dataset))
#             foreign_spectrum = dataset[rand_idx].numpy()
            
#             ### Find all peaks in the random spectrum
#             f_peaks, _ = find_peaks(foreign_spectrum, height=np.mean(foreign_spectrum))
            
#             ### If there is at least one peak select one at random with envelope 
#             if len(f_peaks) > 0:
#                 p_idx = np.random.choice(f_peaks)
#                 #### Calculate envelope width at 0.8 relative length (we want to capture it completely)
#                 widths, width_heights, left_ips, right_ips = peak_widths(
#                     foreign_spectrum, [p_idx], rel_height=0.8
#                 )
                
#                 #### Convert interpolated indices to xs/ys indices
#                 start = int(left_ips[0])
#                 end = int(right_ips[0]) + 1
                
#                 #### Add the peak envelope to the original spectrum
#                 peak_envelope = foreign_spectrum[start:end]
#                 noisy_batch[i, start:end] += torch.from_numpy(peak_envelope).to(device)

#         ## Re-normalize - data consistency 
#         if norm_type == 'TIC':
#             ### Total Ion Count normalization: ensure sum of intensities is constant
#             # original_tic = torch.sum(vec[i])
#             noisy_tic = torch.sum(noisy_batch[i])
#             if noisy_tic > 0:
#                 noisy_batch[i] = noisy_batch[i] / noisy_tic
#         # TODO - implment other normalization        
        
#     return noisy_batch


# TODO - fasten the function
def estimate_max_peak_width(IMSLoader, sample_size=100):
    """
    Analyzes a sample of spectra to find the largest peak envelope width (in bins).
    This helps in selecting the size of the first kernel in the CNN.[cite: 8]
    """
    total_spectra = len(IMSLoader)
    # Randomly select indices of spectra for analysis (for performance efficiency)[cite: 5]
    indices = np.random.choice(total_spectra, min(sample_size, total_spectra), replace=False)
    
    max_width = 0
    
    for idx in indices:
        # Retrieve the normalized spectrum from the Loader
        spectrum = IMSLoader[idx].numpy()
        
        # Find peaks 
        peaks, properties = find_peaks(spectrum, prominence=np.mean(spectrum))
        
        if len(peaks) > 0:
            # Calculate peak widths at 10% of their height (envelope at the base)
            widths = peak_widths(spectrum, peaks, rel_height=0.9)[0]
            if len(widths) > 0:
                current_max = np.max(widths)
                if current_max > max_width:
                    max_width = current_max
                    
    # Return as int, minimum of 3 (to ensure the kernel is meaningful)
    # Round up to the nearest odd number for kernel symmetry
    suggested_kernel = int(np.ceil(max_width))
    if suggested_kernel % 2 == 0:
        suggested_kernel += 1
        
    return max(3, suggested_kernel)


def precompute_peak_bank(dataset: IMSPyTorchDataset, max_peaks_per_spectrum: int = 2):
    """Pre-identifies a fixed number of peak envelopes across the dataset.
    This creates a diverse noise bank without exhausting system memory."""
    peak_bank = []
    
    print(f"Building noise bank (max {max_peaks_per_spectrum} peaks per spectrum)...")
    
    for i in range(len(dataset)):
        ## Load spectrum to CPU
        spectrum_np = dataset[i].numpy()
        
        ## Find peaks (scipy library) 
        peaks, _ = find_peaks(spectrum_np, height=np.mean(spectrum_np))
        
        if len(peaks) > 0:
            ## Limit the number of peaks stored per spectrum to save memory
            selected_peaks = np.random.choice(
                peaks, 
                size=min(len(peaks), max_peaks_per_spectrum), 
                replace=False
            )
            
            for p_idx in selected_peaks:
                ### Calculate envelope width using original parameters
                widths, _, left_ips, right_ips = peak_widths(
                    spectrum_np, [p_idx], rel_height=0.8
                )
                
                start = int(left_ips[0])
                end = int(right_ips[0]) + 1
                
                ### Store as small torch float16/32 tensors to save space
                peak_vals = torch.from_numpy(spectrum_np[start:end]).float()
                peak_bank.append((start, end, peak_vals))
                
    ## Attach the bank to the dataset object
    print(f"PeakBank created with {len(peak_bank)} total noise samples.")
    return peak_bank