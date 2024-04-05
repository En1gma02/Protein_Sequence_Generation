import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

print("Using device:", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Define the dataset class
class ProteinDataset(Dataset):
    def __init__(self, data, token_to_index, max_seq_length):
        self.data = data
        self.token_to_index = token_to_index
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx]['sequence']
        sequence = [self.token_to_index.get(token, 0) for token in
                    sequence[:self.max_seq_length]]  # Truncate long sequences
        return sequence


# Define the RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x, lengths, hidden):
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn(packed, hidden)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(2 * 2, batch_size, self.hidden_size).to(device),  # Bidirectional and 2 layers
                torch.zeros(2 * 2, batch_size, self.hidden_size).to(device))


# Load and preprocess data
print("Loading data...")
data = pd.read_csv('E:/Hackathons/IIT BHU/Protien Sequencing/preprocessed_encoded_vectorized.csv')
all_amino_acids = set(''.join(data['sequence']))
token_to_index = {token: idx + 1 for idx, token in enumerate(sorted(all_amino_acids))}  # Reserve 0 for padding
input_size = len(token_to_index) + 1  # Add 1 for padding token
max_seq_length = 500  # Maximum sequence length
hidden_size = 256  # Hidden size of the RNN
output_size = len(token_to_index) + 1  # Same as input size
num_layers = 2  # Number of RNN layers
dropout = 0.2  # Dropout rate

# Split data into train and test sets
print("Splitting data into train and test sets...")
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Create datasets and dataloaders
print("Creating datasets and dataloaders...")
train_dataset = ProteinDataset(train_data, token_to_index, max_seq_length)
test_dataset = ProteinDataset(test_data, token_to_index, max_seq_length)

# Define the collate function for DataLoader
def collate_fn(batch):
    # Sort batch by sequence length (required for pack_padded_sequence)
    batch.sort(key=lambda x: len(x), reverse=True)
    # Convert list of lists to a list of tensors
    sequences = [torch.tensor(item) for item in batch]
    # Pad sequences
    sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    return sequences

# Create DataLoader instances with the updated collate_fn
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

# Initialize the model
print("Initializing the model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RNNModel(input_size, hidden_size, output_size, num_layers, dropout).to(device)

# Define loss function and optimizer
print("Defining loss function and optimizer...")
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, 'min')  # Reduce LR on plateau

# Training loop
num_epochs = 3
best_val_loss = np.inf
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    print(f"Epoch {epoch+1} started.")
    for batch_idx, batch in enumerate(train_loader):
        print(f"Processing batch {batch_idx+1}/{len(train_loader)}...")
        sequences = batch[:, :-1]
        lengths = (sequences != 0).sum(dim=1)
        targets = batch[:, 1:]
        optimizer.zero_grad()
        hidden = model.init_hidden(batch.size(0))
        output, _ = model(sequences, lengths, hidden)
        loss = criterion(output.contiguous().view(-1, output.shape[-1]), targets.contiguous().view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Evaluate on validation set
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            sequences = batch[:, :-1]
            lengths = (sequences != 0).sum(dim=1)
            targets = batch[:, 1:]
            hidden = model.init_hidden(batch.size(0))
            output, _ = model(sequences, lengths, hidden)
            loss = criterion(output.contiguous().view(-1, output.shape[-1]), targets.contiguous().view(-1))
            val_loss += loss.item()

    val_loss /= len(test_loader)
    scheduler.step(val_loss)

    print(f"Epoch {epoch + 1}, Train Loss: {total_loss}, Val Loss: {val_loss}")

    # Save the model if validation loss has decreased
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'trained_model_gpt.pth')
