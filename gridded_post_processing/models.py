"""
The two neural network architectures used for gridded post-processing.

Both take one 6x6 degree patch of raw forecast (flattened to a vector), the day
of year and the lead time, and predict the forecast ERROR at every grid cell
of the patch. The corrected forecast is the raw forecast plus that prediction.

    MultilayerPerceptron  fully connected layers on the flattened patch (the
                          paper's model)
    UNet                  a convolutional encoder-decoder with skip
                          connections, for the architecture comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultilayerPerceptron(nn.Module):
    """
    Fully connected network that predicts the forecast error of a patch.

    The flattened patch is concatenated with the sine and cosine of the day of
    year and, when there is more than one lead time, a learned embedding of the
    lead time, then passed through num_hidden_layers ReLU layers of width
    hidden_dim (each followed by dropout) and a linear output layer.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers, n_lead_times,
                 lead_time_embedding_dim, dropout_rate):
        """
        Build the layers.

        Inputs:
            input_dim (int): length of the flattened input patch.
            hidden_dim (int): width of every hidden layer.
            output_dim (int): length of the flattened output patch.
            num_hidden_layers (int): number of hidden layers.
            n_lead_times (int): number of lead times the network is trained on.
                With 1, no lead time embedding is used.
            lead_time_embedding_dim (int): length of the lead time embedding.
            dropout_rate (float): dropout after every hidden layer (0 disables).

        Returns:
            None.
        """
        super().__init__()

        # The day of year adds 2 inputs (sine and cosine); the lead time
        # embedding adds its length
        network_input_dim = input_dim + 2
        self.lead_time_embedding = None
        if n_lead_times > 1:
            self.lead_time_embedding = nn.Embedding(n_lead_times, lead_time_embedding_dim)
            network_input_dim += lead_time_embedding_dim

        layers = []
        layer_input_dim = network_input_dim
        for _ in range(num_hidden_layers):
            layers += [nn.Linear(layer_input_dim, hidden_dim), nn.ReLU()]
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            layer_input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, forecast, lead_time_index, day_of_year_features):
        """
        Predict the forecast error.

        Inputs:
            forecast (torch.Tensor): (batch, input_dim) normalised forecast patches.
            lead_time_index (torch.Tensor): (batch,) integer lead time positions.
            day_of_year_features (torch.Tensor): (batch, 2) sine and cosine of
                the day of year.

        Returns:
            torch.Tensor of shape (batch, output_dim): the predicted error.
        """
        network_input = torch.cat([forecast, day_of_year_features], dim=-1)
        if self.lead_time_embedding is not None:
            network_input = torch.cat([network_input, self.lead_time_embedding(lead_time_index)],
                                      dim=-1)
        return self.net(network_input)


class UNet(nn.Module):
    """
    Convolutional encoder-decoder that predicts the forecast error of a patch.

    The day of year (and the lead time embedding, with more than one lead
    time) is broadcast over the patch and added as extra input channels. The
    encoder halves the patch size at each level down to at least 4x4 (at most
    5 levels), doubling the channels each level up to a cap of 128; the decoder
    mirrors it, joining each level to the encoder output of the same size.
    """

    def __init__(self, hidden_dim, n_latitudes, n_longitudes, n_lead_times,
                 lead_time_embedding_dim, dropout_rate):
        """
        Build the layers.

        Inputs:
            hidden_dim (int): channels of the first encoder level.
            n_latitudes, n_longitudes (int): patch size in grid cells.
            n_lead_times (int): number of lead times the network is trained on.
            lead_time_embedding_dim (int): length of the lead time embedding.
            dropout_rate (float): channel dropout inside every convolution block.

        Returns:
            None.
        """
        super().__init__()
        self.height = n_latitudes
        self.width = n_longitudes

        # Conditioning channels: day of year sine and cosine, plus the lead
        # time embedding when there is more than one lead time
        self.conditioning_channels = 2
        self.lead_time_embedding = None
        if n_lead_times > 1:
            self.lead_time_embedding = nn.Embedding(n_lead_times, lead_time_embedding_dim)
            self.conditioning_channels += lead_time_embedding_dim

        # Number of levels: one more than the number of times the patch can be
        # halved while staying at least 4 cells wide, capped at 5
        number_of_halvings = 0
        size = min(n_latitudes, n_longitudes)
        while size >= 4:
            number_of_halvings += 1
            size //= 2
        self.num_levels = min(number_of_halvings + 1, 5)

        # Encoder: one convolution block per level, with pooling between levels
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.encoder_channels = []
        input_channels = 1 + self.conditioning_channels
        output_channels = hidden_dim
        for level in range(self.num_levels):
            self.encoders.append(self.convolution_block(input_channels, output_channels,
                                                        dropout_rate))
            self.encoder_channels.append(output_channels)
            if level < self.num_levels - 1:
                self.pools.append(nn.MaxPool2d(kernel_size=2))
            input_channels = output_channels
            output_channels = min(output_channels * 2, 128)

        # Decoder: upsample, join with the matching encoder output, convolve
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        for level in range(self.num_levels - 1):
            deeper_channels = self.encoder_channels[self.num_levels - 1 - level]
            skip_channels = self.encoder_channels[self.num_levels - 2 - level]
            self.upconvs.append(nn.ConvTranspose2d(deeper_channels, skip_channels,
                                                   kernel_size=2, stride=2))
            self.decoders.append(self.convolution_block(2 * skip_channels, skip_channels,
                                                        dropout_rate))

        self.final_conv = nn.Conv2d(self.encoder_channels[0], 1, kernel_size=1)

    @staticmethod
    def convolution_block(input_channels, output_channels, dropout_rate):
        """
        Two 3x3 convolutions, each followed by batch normalisation, ReLU and dropout.

        Inputs:
            input_channels, output_channels (int): channel counts.
            dropout_rate (float): channel dropout rate.

        Returns:
            nn.Sequential.
        """
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
        )

    def forward(self, forecast, lead_time_index, day_of_year_features):
        """
        Predict the forecast error.

        Inputs:
            forecast (torch.Tensor): (batch, n_latitudes * n_longitudes)
                normalised forecast patches.
            lead_time_index (torch.Tensor): (batch,) integer lead time positions.
            day_of_year_features (torch.Tensor): (batch, 2) sine and cosine of
                the day of year.

        Returns:
            torch.Tensor of shape (batch, n_latitudes * n_longitudes): the
            predicted error.
        """
        batch_size = forecast.shape[0]
        patch = forecast.view(batch_size, 1, self.height, self.width)

        # Broadcast the conditioning over the patch and add it as extra channels
        conditioning = day_of_year_features
        if self.lead_time_embedding is not None:
            conditioning = torch.cat([conditioning, self.lead_time_embedding(lead_time_index)],
                                     dim=1)
        conditioning = conditioning.view(batch_size, self.conditioning_channels, 1, 1).expand(
            batch_size, self.conditioning_channels, self.height, self.width)
        features = torch.cat([patch, conditioning], dim=1)

        # Encoder, keeping each level's output for the skip connections
        encoder_outputs = []
        for level in range(self.num_levels):
            features = self.encoders[level](features)
            if level < self.num_levels - 1:
                encoder_outputs.append(features)
                features = self.pools[level](features)

        # Decoder. Pooling an odd size loses a row, so the skip connection is
        # centre-cropped to the upsampled size when the two differ.
        for level in range(len(self.upconvs)):
            features = self.upconvs[level](features)
            skip = encoder_outputs[-(level + 1)]
            if features.shape[2:] != skip.shape[2:]:
                row_start = (skip.shape[2] - features.shape[2]) // 2
                column_start = (skip.shape[3] - features.shape[3]) // 2
                skip = skip[:, :, row_start:row_start + features.shape[2],
                            column_start:column_start + features.shape[3]]
            features = self.decoders[level](torch.cat([features, skip], dim=1))
        features = self.final_conv(features)

        # Pad back to the patch size if pooling an odd size made the output smaller
        missing_rows = self.height - features.shape[2]
        missing_columns = self.width - features.shape[3]
        if missing_rows or missing_columns:
            features = F.pad(features, (missing_columns // 2,
                                        missing_columns - missing_columns // 2,
                                        missing_rows // 2, missing_rows - missing_rows // 2),
                             mode="replicate")
        return features.view(batch_size, -1)
