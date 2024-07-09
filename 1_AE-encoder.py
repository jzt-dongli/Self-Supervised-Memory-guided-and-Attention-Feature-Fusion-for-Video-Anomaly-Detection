import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import os.path
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

class Encoder(nn.Module):
    def __init__(self, input_dim=512, encoded_dim=34):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, encoded_dim)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, encoded_dim=34, output_dim=512):
        super(Decoder, self).__init__()

        self.linear1 = nn.Linear(encoded_dim, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, output_dim)

    def forward(self, x):

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(input_dim=512, encoded_dim=34).to(device)
    decoder = Decoder(encoded_dim=34, output_dim=512).to(device)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-2)
    criterion = nn.MSELoss()

    file_path = 'your_deep_features_train.npy'
    data = np.load(file_path, allow_pickle=True)
    test_file_path = 'your_deep_features_test.npy'
    test_data = np.load(test_file_path, allow_pickle=True)

    stacked_data = np.vstack(data)
    data_tensor = torch.tensor(stacked_data, dtype=torch.float32)
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=4096, shuffle=True)
    test_stacked_data = np.vstack(test_data)
    test_data_tensor = torch.tensor(test_stacked_data, dtype=torch.float32)
    test_dataset = TensorDataset(test_data_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=4096)

    if os.path.isfile('state/ae_encoder/encoder_best_epoch.pth'):
        encoder.load_state_dict(torch.load('state/ae_encoder/encoder_best_epoch.pth'))
    if os.path.isfile('state/ae_encoder/decoder_best_epoch.pth'):
        decoder.load_state_dict(torch.load('state/ae_encoder/decoder_best_epoch.pth'))

    num_epochs = 1000
    best_loss = float('inf')
    train_losses = []
    test_losses = []

    for epoch in tqdm(range(num_epochs)):

        encoder.train()
        decoder.train()
        ae_total_loss = 0
        for inputs, in dataloader:
            inputs = inputs.to(device)


            encoded = encoder(inputs)
            decoded = decoder(encoded)


            loss = criterion(decoded, inputs)
            ae_total_loss += loss.item()


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        average_train_loss = ae_total_loss / len(dataloader)
        train_losses.append(average_train_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {average_train_loss:.4f}')


        encoder.eval()
        decoder.eval()
        ae_test_loss = 0
        with torch.no_grad():
            for test_inputs, in test_dataloader:
                test_inputs = test_inputs.to(device)


                encoded = encoder(test_inputs)
                decoded = decoder(encoded)


                loss = criterion(decoded, test_inputs)
                ae_test_loss += loss.item()

        average_test_loss = ae_test_loss / len(test_dataloader)
        test_losses.append(average_test_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Test Loss: {average_test_loss:.8f}')

        # 检查是否有新的最低损失
        if average_test_loss < best_loss:
            best_loss = average_test_loss
            torch.save(encoder.state_dict(), f'state/ae_encoder/encoder_best_epoch.pth')
            torch.save(decoder.state_dict(), f'state/ae_encoder/decoder_best_epoch.pth')
            print(f'New best test loss {best_loss:.8f}, model saved.')
