import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io as sio
import time
import os
import matplotlib.pyplot as plt

# --------------------
# 1) Custom NMSE Loss
# --------------------
class NMSELoss(nn.Module):
    """
    NMSE = ||y_pred - y_true||^2 / ||y_true||^2
    """
    def __init__(self, eps=1e-12):
        super(NMSELoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        # y_pred, y_true shape: (batch, 2, H, W) for complex: channel=2 => real/imag
        num = torch.sum((y_pred - y_true)**2, dim=[1,2,3])
        den = torch.sum(y_true**2, dim=[1,2,3]) + self.eps
        nmse_batch = num / den
        # Return average NMSE across batch
        return torch.mean(nmse_batch)


# --------------------
# 2) The DnCNN Multi-Block Model
# --------------------
class DnCNN_MultiBlock(nn.Module):
    """
    Mimics the Keras DnCNN_MultiBlock:
      - B blocks
      - Each block:
         * (depth-1) layers of Conv + BN + ReLU
         * final Conv
         * subtract from input
      - Output is the final 'cleaned' image
    """
    def __init__(self, block=3, depth=16, image_channels=2, filters=64, use_bnorm=True):
        super(DnCNN_MultiBlock, self).__init__()
        self.block = block
        self.depth = depth
        self.image_channels = image_channels
        self.filters = filters
        self.use_bnorm = use_bnorm

        # Each "block" is a sequence of (depth-1) Conv+BN+ReLU, then a final Conv
        # We'll store each block in a ModuleList
        self.blocks = nn.ModuleList()
        for b in range(block):
            layers = []
            for i in range(depth - 1):
                in_ch = image_channels if i == 0 else filters
                conv = nn.Conv2d(in_channels=in_ch,
                                 out_channels=filters,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 bias=False)
                layers.append(conv)
                if use_bnorm:
                    layers.append(nn.BatchNorm2d(filters, momentum=0.0, eps=1e-4))
                layers.append(nn.ReLU(inplace=True))

            # Final conv of the block
            layers.append(
                nn.Conv2d(in_channels=filters,
                          out_channels=image_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False)
            )
            self.blocks.append(nn.Sequential(*layers))

    def forward(self, x):
        # Repeatedly apply each block, then x = x - residual
        for block_module in self.blocks:
            residual = block_module(x)
            x = x - residual
        return x


# --------------------
# 3) Training Function
# --------------------
def train_model(model, 
                train_data, 
                test_data, 
                epochs=400, 
                batch_size=64, 
                lr=1e-3,
                early_stop_patience=5,
                device='cuda'):
    """
    train_data: (x_train, y_train) 
    test_data:  (x_test, y_test)
    
    x_train shape is (N, H, W, 2) => we'll convert to (N, 2, H, W) in PyTorch
    """

    # Unpack
    x_train, y_train = train_data
    x_test,  y_test  = test_data

    # Move to (N, 2, H, W) for PyTorch
    # If data is float64 from MATLAB, convert to float32
    x_train = np.transpose(x_train, (0,3,1,2)).astype('float32')
    y_train = np.transpose(y_train, (0,3,1,2)).astype('float32')
    x_test  = np.transpose(x_test,  (0,3,1,2)).astype('float32')
    y_test  = np.transpose(y_test,  (0,3,1,2)).astype('float32')

    # Convert numpy to torch tensors
    x_train_torch = torch.from_numpy(x_train)
    y_train_torch = torch.from_numpy(y_train)
    x_test_torch  = torch.from_numpy(x_test)
    y_test_torch  = torch.from_numpy(y_test)

    # Create Datasets and Loaders
    train_dataset = torch.utils.data.TensorDataset(x_train_torch, y_train_torch)
    test_dataset  = torch.utils.data.TensorDataset(x_test_torch,  y_test_torch)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False)
    test_loader  = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               drop_last=False)

    print('Train size:', len(train_dataset), ' Test size:', len(test_dataset))

    # Model, optimizer, loss
    model = model.to(device)
    criterion = NMSELoss()  # Our custom NMSE loss
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # For (simple) early stopping tracking
    best_val_loss = float('inf')
    patience_counter = 0

    # Lists for plotting
    train_losses = []
    val_losses = []

    # -------------
    # Training Loop
    # -------------
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs  = inputs.to(device)   # (batch, 2, H, W)
            targets = targets.to(device)  # (batch, 2, H, W)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # -------------
        # Validation
        # -------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(test_loader):
                inputs  = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        epoch_val_loss = val_loss / len(test_loader)
        val_losses.append(epoch_val_loss)

        print(f"Epoch [{epoch+1}/{epochs}], Train NMSE: {epoch_train_loss:.6f}, Val NMSE: {epoch_val_loss:.6f}")

        # -------------
        # Early Stopping
        # -------------
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print("Early stopping triggered!")
                break

    # Load best model
    model.load_state_dict(torch.load("best_model.pth"))

    # Plot training curve
    plt.figure()
    plt.plot(range(len(train_losses)), train_losses, 'g', label='Train NMSE')
    plt.plot(range(len(val_losses)),   val_losses,   'r', label='Val NMSE')
    plt.xlabel('Epoch')
    plt.ylabel('NMSE Loss')
    plt.legend()
    plt.title('Training and Validation NMSE')
    plt.savefig('loss_curve.png')
    plt.show()

    return model


# --------------------
# 4) Main (Example)
# --------------------
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 64
    epochs = 400

    # Instantiate model
    model = DnCNN_MultiBlock(
        block=3,          # B=3 blocks
        depth=16,         # each block has (depth-1) conv+bn+relu + 1 final conv
        image_channels=2, # real+imag => 2 channels
        filters=64,
        use_bnorm=True
    )
    print(model)

    # -----------
    # Load data (example)
    # -----------
    # Make sure your .mat keys match these.
    data_xtr   = sio.loadmat('x_train_Rician_CSCG_K10dB_60000_M8_N32_5dB.mat')
    data_ytr   = sio.loadmat('y_train_Rician_CSCG_K10dB_60000_M8_N32_5dB.mat')
    data_xtest = sio.loadmat('x_test_Rician_CSCG_K10dB_20000_M8_N32_5dB.mat')
    data_ytest = sio.loadmat('y_test_Rician_CSCG_K10dB_20000_M8_N32_5dB.mat')

    x_train = data_xtr['x_train']  # shape (60000, 8, 8, 2) in MATLAB
    y_train = data_ytr['y_train']  # shape (60000, 8, 8, 2) in MATLAB
    x_test  = data_xtest['x_test'] # shape (20000, 8, 8, 2)
    y_test  = data_ytest['y_test'] # shape (20000, 8, 8, 2)

    # -----------
    # Train
    # -----------
    trained_model = train_model(
        model,
        train_data=(x_train, y_train),
        test_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        lr=1e-3,
        early_stop_patience=5,
        device=device
    )

    # -----------
    # Evaluate & Predict
    # -----------
    trained_model.eval()
    x_test_torch = torch.from_numpy(
        np.transpose(x_test.astype('float32'), (0,3,1,2))
    ).to(device)

    # Example: compute inference time on test set
    start_time = time.time()
    with torch.no_grad():
        # Forward in batches
        predictions = []
        batch_size_eval = 64
        for i in range(0, x_test_torch.shape[0], batch_size_eval):
            batch_in = x_test_torch[i:i+batch_size_eval]
            batch_out = trained_model(batch_in)
            # batch_out shape: (bs, 2, 8, 8)
            predictions.append(batch_out.cpu().numpy())
    end_time = time.time()

    predictions = np.concatenate(predictions, axis=0)
    # If you want "residual" = input - predicted, as in the Keras code:
    # recall input shape after transpose is (N,2,H,W)
    # so we revert back to (N,H,W,2) for saving
    x_test_np = x_test_torch.cpu().numpy()  # shape (N,2,H,W)
    residual_np = x_test_np - predictions   # shape (N,2,H,W)
    # rearrange to (N,H,W,2)
    residual_np = np.transpose(residual_np, (0, 2, 3, 1))

    # Save to .mat
    sio.savemat('Model_CDRN_20dB_Residual_Prediction210106_pytorch.mat', 
                {'data': residual_np})
    print('Prediction saved to .mat file')

    # Per-sample inference time (approx)
    print("Inference time per sample: {:.6f}s".format(
        (end_time - start_time) / x_test_torch.shape[0])
    )


if __name__ == "__main__":
    main()
