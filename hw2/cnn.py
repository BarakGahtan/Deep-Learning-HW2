import torch
import itertools as it
import torch.nn as nn


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(CONV -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """

    def __init__(self, in_size, out_classes: int, channels: list,
                 pool_every: int, hidden_dims: list):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [(CONV -> ReLU)*P -> MaxPool]*(N/P)
        #  Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        #  Pooling to reduce dimensions after every P convolutions.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ReLUs should exist at the end, without a MaxPool after them.
        # ====== YOUR CODE: ======
        N = len(self.channels)
        P = self.pool_every
        isAnotherLayer = N % P != 0

        num_max_polling = N//P
        input_channels = in_channels
        conv_count = 0
        for _ in range(num_max_polling):
            for _ in range(P):
                layers.append(torch.nn.Conv2d(input_channels,
                                              self.channels[conv_count], 3, padding=1))
                layers.append(torch.nn.ReLU())
                input_channels = self.channels[conv_count]
                conv_count += 1
            layers.append(torch.nn.MaxPool2d(2))
        if isAnotherLayer:
            for i in range(N-conv_count):
                layers.append(torch.nn.Conv2d(input_channels,
                                              self.channels[i+conv_count], 3, padding=1))
                layers.append(torch.nn.ReLU())
                # layers.append(blocks.ReLU())
                input_channels = self.channels[i+conv_count]

        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        #  (Linear -> ReLU)*M -> Linear
        #  You'll first need to calculate the number of features going in to
        #  the first linear layer.
        #  The last Linear layer should have an output dim of out_classes.
        # ====== YOUR CODE: ======
        N = len(self.channels)
        P = self.pool_every
        # print("P = ", P)
        # print("N = ", N)
        h_out, w_out = 0, 0
        num_max_pool = N//P
        h_in = in_h
        w_in = in_w
        for _ in range(num_max_pool):
            # for _ in range(P):
            #     # h_out = h_in - 2
            #     # w_out = w_in - 2
            #     h_out = h_in
            #     w_out = w_in
            #     h_in, w_in = h_out, w_out
            h_out = int((h_in - 2)/2.0 + 1.0)
            w_out = int((w_in - 2)/2.0 + 1.0)
            h_in, w_in = h_out, w_out
        # for _ in range(N % P):
            # h_out = h_in - 2
            # w_out = w_in - 2
            # h_out = h_in
            # w_out = w_in
            # h_in, w_in = h_out, w_out

        # print("h_out = ", h_out)
        # print("w_out = ", w_out)

        last_channel_size = self.channels[-1]
        # print("last_channel_size = ", last_channel_size)

        in_linear_features = h_out*w_out*last_channel_size
        for next_dimention in self.hidden_dims:
            # print(next_dimention)
            layers.append(torch.nn.Linear(
                in_linear_features, next_dimention))
            layers.append(torch.nn.ReLU())
            in_linear_features = next_dimention
        layers.append(torch.nn.Linear(
            self.hidden_dims[-1], self.out_classes))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        # ====== YOUR CODE: ======
        extracted_features = self.feature_extractor(x)
        extracted_features = extracted_features.view(extracted_features.size(0), -1)
        out = self.classifier(extracted_features)
        # ========================
        return out


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(self, in_channels: int, channels: list, kernel_sizes: list,
                 batchnorm=False, dropout=0.):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
        convolution in the block. The length determines the number of
        convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
        be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
        convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
        Zero means don't apply dropout.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        self.main_path, self.shortcut_path = None, None

        # TODO: Implement a generic residual block.
        #  Use the given arguments to create two nn.Sequentials:
        #  the main_path, which should contain the convolution, dropout,
        #  batchnorm, relu sequences, and the shortcut_path which should
        #  represent the skip-connection.
        #  Use convolutions which preserve the spatial extent of the input.
        #  For simplicity of implementation, we'll assume kernel sizes are odd.
        # ====== YOUR CODE: ======
        shortcut, layers_main = [], []
        prev_channels = in_channels
        for i in range(len(channels)):
            layers_main.append(torch.nn.Conv2d(
                prev_channels, channels[i], kernel_size=kernel_sizes[i], padding=int(0.5*(kernel_sizes[i]-1))))
            if i < (len(channels)-1):
                if(batchnorm):
                    layers_main.append(torch.nn.BatchNorm2d(channels[i]))
                if(dropout > 0):
                    layers_main.append(torch.nn.Dropout(dropout))
                layers_main.append(torch.nn.ReLU())
            prev_channels = channels[i]

        last_channel = channels[-1]
        if last_channel != in_channels:
            shortcut.append(torch.nn.Conv2d(
                in_channels, last_channel, kernel_size=1, padding=0, bias=False))
        # if(batchnorm):
        #     shortcut.append(torch.nn.BatchNorm2d(channels[i]))
        # if(dropout > 0):
        #     shortcut.append(torch.nn.Dropout(dropout))

        self.main_path = nn.Sequential(*layers_main)
        self.shortcut_path = nn.Sequential(*shortcut)
        # ========================

    def forward(self, x):
        out = self.main_path(x)
        out += self.shortcut_path(x)
        out = torch.relu(out)
        return out


class ResNetClassifier(ConvClassifier):
    def __init__(self, in_size, out_classes, channels, pool_every,
                 hidden_dims):
        super().__init__(in_size, out_classes, channels, pool_every,
                         hidden_dims)

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ReLU)*P -> MaxPool]*(N/P)
        #   \------- SKIP ------/
        #  Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        #  Pooling to reduce dimensions after every P convolutions.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ReLUs (with a skip over them) should exist at the end,
        #  without a MaxPool after them.
        # ====== YOUR CODE: ======
        N = len(self.channels)
        P = self.pool_every
        current_channel = in_channels
        current_channel_index = 0
        for _ in range(N//P):
            curr_channels = self.channels[current_channel_index:(
                current_channel_index+P)]  # up until current_channel_index+P-1
            layers.append(ResidualBlock(
                in_channels=current_channel, channels=curr_channels, kernel_sizes=[3]*P))
            current_channel_index += P
            current_channel = self.channels[current_channel_index-1]
            layers.append(torch.nn.MaxPool2d(2, padding=0))
        if N % P != 0:
            layers.append(ResidualBlock(
                in_channels=current_channel, channels=self.channels[-(N % P):], kernel_sizes=[3]*(N % P)))
        # ========================
        seq = nn.Sequential(*layers)
        return seq


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, channels, pool_every,
                 hidden_dims):
        super().__init__(in_size, out_classes, channels, pool_every,
                         hidden_dims)

    # TODO: Change whatever you want about the ConvClassifier to try to
    #  improve it's results on CIFAR-10.
    #  For example, add batchnorm, dropout, skip connections, change conv
    #  filter sizes etc.
    # ====== YOUR CODE: ======
    class skipLayer(nn.Module):
        def __init__(self, in_channel, out_channel, kernel_size, batchnorm=False, dropout=0.0):
            super().__init__()
            self.normaBlock = nn.Sequential(
                torch.nn.Conv2d(in_channel, out_channel[0], kernel_size=kernel_size[0],
                                padding=int(0.5*(kernel_size[0]-1))),
                torch.nn.BatchNorm2d(out_channel[0]),
                torch.nn.Dropout(dropout)
            )
            self.sortcutBlock = nn.Sequential()
            if in_channel != out_channel[0]:
                self.sortcutBlock = nn.Sequential(
                    torch.nn.Conv2d(in_channel, out_channel[0], kernel_size=1,
                                    padding=0)
                )

        def forward(self, x):
            out = self.normaBlock(x)
            out += self.sortcutBlock(x)
            out = torch.relu(out)
            return out

    def _make_feature_extractor(self):
        in_channels, _,_ = tuple(self.in_size)
        layers = []
        kernels = [3,3,3,3,3,3,5,5,5,7,7]
        count=1
        prev_channel = in_channels
        for channel in self.channels:
            if(count==len(self.channels)):
                break
            random_kernel_size = kernels[torch.randint(0,len(kernels)-1,(1,)).item()]
            layers.append(self.skipLayer(prev_channel, [channel],[random_kernel_size],False,0.5))
            if count % self.pool_every==0:
                layers.append(torch.nn.MaxPool2d(2))
            count+=1
            prev_channel = channel
        layers.append(torch.nn.Conv2d(self.channels[-2],self.channels[-1] ,kernel_size=3,padding=1))
        if len(self.channels)%self.pool_every==0:
            layers.append(torch.nn.MaxPool2d(2))


        seq = nn.Sequential(*layers)
        # print(seq)
        return seq
    # def _make_classifier(self):




    # ========================
