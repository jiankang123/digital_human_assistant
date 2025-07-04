import torch
import torch.nn as nn
import torch.nn.functional as F


class DSDilatedConv1d(nn.Module):
    """Dilated Depthwise-Separable Convolution"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        stride: int = 1,
        bias: bool = True,
    ):
        super(DSDilatedConv1d, self).__init__()
        self.receptive_fields = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            padding=0,
            dilation=dilation,
            stride=stride,
            groups=in_channels,
            bias=bias,
        )
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.pointwise = nn.Conv1d(in_channels,
                                   out_channels,
                                   kernel_size=1,
                                   padding=0,
                                   dilation=1,
                                   bias=bias)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, inputs: torch.Tensor):
        outputs = self.conv(inputs)
        outputs = self.bn1(outputs)
        outputs = F.relu(outputs)
        outputs = self.pointwise(outputs)
        outputs = self.bn2(outputs)
        outputs = F.relu(outputs)
        return outputs


class TCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        res_channels: int,
        kernel_size: int,
        dilation: int,
        causal: bool,
    ):
        super(TCNBlock, self).__init__()
        self.in_channels = in_channels
        self.res_channels = res_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.causal = causal
        self.receptive_fields = dilation * (kernel_size - 1)
        self.half_receptive_fields = self.receptive_fields // 2
        self.conv1 = DSDilatedConv1d(
            in_channels=in_channels,
            out_channels=res_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(res_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=res_channels,
                               out_channels=res_channels,
                               kernel_size=1)
        self.bn2 = nn.BatchNorm1d(res_channels)
        self.relu2 = nn.ReLU()

    def forward(self, inputs: torch.Tensor):
        outputs = self.conv1(inputs)
        outputs = self.bn2(self.conv2(outputs))
        if self.causal:
            inputs = inputs[:, :, self.receptive_fields:]
        else:
            inputs = inputs[:, :, self.
                            half_receptive_fields:-self.half_receptive_fields]
        if self.in_channels == self.res_channels:
            res_out = self.relu2(outputs + inputs)
        else:
            res_out = self.relu2(outputs)
        return res_out


class TCNStack(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stack_num: int,
        stack_size: int,
        res_channels: int,
        kernel_size: int,
        causal: bool,
    ):
        super(TCNStack, self).__init__()
        self.in_channels = in_channels
        self.stack_num = stack_num
        self.stack_size = stack_size
        self.res_channels = res_channels
        self.kernel_size = kernel_size
        self.causal = causal
        self.res_blocks = self.stack_tcn_blocks()
        self.receptive_fields = self.calculate_receptive_fields()
        self.res_blocks = nn.Sequential(*self.res_blocks)

    def calculate_receptive_fields(self):
        receptive_fields = 0
        for block in self.res_blocks:
            receptive_fields += block.receptive_fields
        return receptive_fields

    def build_dilations(self):
        dilations = []
        for s in range(0, self.stack_size):
            for l in range(0, self.stack_num):
                dilations.append(2**l)
        return dilations

    def stack_tcn_blocks(self):
        dilations = self.build_dilations()
        res_blocks = nn.ModuleList()

        res_blocks.append(
            TCNBlock(
                self.in_channels,
                self.res_channels,
                self.kernel_size,
                dilations[0],
                self.causal,
            ))
        for dilation in dilations[1:]:
            res_blocks.append(
                TCNBlock(
                    self.res_channels,
                    self.res_channels,
                    self.kernel_size,
                    dilation,
                    self.causal,
                ))
        return res_blocks

    def forward(self, inputs: torch.Tensor):
        outputs = inputs
        outputs = self.res_blocks(outputs)
        return outputs


class MDTC(nn.Module):
    """Multi-scale Depthwise Temporal Convolution (MDTC).
    In MDTC, stacked depthwise one-dimensional (1-D) convolution with
    dilated connections is adopted to efficiently model long-range
    dependency of speech. With a large receptive field while
    keeping a small number of model parameters, the structure
    can model temporal context of speech effectively. It aslo
    extracts multi-scale features from different hidden layers
    of MDTC with different receptive fields.
    """
    def __init__(
        self,
        stack_num: int,
        stack_size: int,
        in_channels: int,
        res_channels: int,
        kernel_size: int,
        causal: bool,
        num_classes: int
    ):
        super(MDTC, self).__init__()
        assert kernel_size % 2 == 1
        self.kernel_size = kernel_size
        self.causal = causal
        self.preprocessor = TCNBlock(in_channels,
                                     res_channels,
                                     kernel_size,
                                     dilation=1,
                                     causal=causal)
        self.relu = nn.ReLU()
        self.blocks = nn.ModuleList()
        self.receptive_fields = self.preprocessor.receptive_fields
        for i in range(stack_num):
            self.blocks.append(
                TCNStack(res_channels, stack_size, 1, res_channels,
                         kernel_size, causal))
            self.receptive_fields += self.blocks[-1].receptive_fields
        self.half_receptive_fields = self.receptive_fields // 2
        print('Receptive Fields: %d' % self.receptive_fields)
        self.dense = nn.Linear(res_channels, num_classes)

    def forward(self, x: torch.Tensor):
        if self.causal:
            outputs = F.pad(x, (0, 0, self.receptive_fields, 0, 0, 0),
                            'constant')
        else:
            outputs = F.pad(
                x,
                (0, 0, self.half_receptive_fields, self.half_receptive_fields,
                 0, 0),
                'constant',
            )
        outputs = outputs.transpose(1, 2)
        outputs_list = []
        outputs = self.relu(self.preprocessor(outputs))
        for block in self.blocks:
            outputs = block(outputs)
            outputs_list.append(outputs)

        normalized_outputs = []
        output_size = outputs_list[-1].shape[-1]
        for x in outputs_list:
            remove_length = x.shape[-1] - output_size
            if self.causal and remove_length > 0:
                normalized_outputs.append(x[:, :, remove_length:])
            elif not self.causal and remove_length > 1:
                half_remove_length = remove_length // 2
                normalized_outputs.append(
                    x[:, :, half_remove_length:-half_remove_length]
                )
            else:
                normalized_outputs.append(x)

        outputs = torch.zeros_like(outputs_list[-1], dtype=outputs_list[-1].dtype)
        for x in normalized_outputs:
            outputs += x
        outputs = outputs.transpose(1, 2)
        outputs = self.dense(outputs)
        outputs = F.softmax(outputs, -1)
        return outputs
    
    def receptive_field(self, ):
        return self.receptive_fields

    def paramter_nums(self, ):
        return sum(p.numel() for p in self.parameters())

if __name__ == '__main__':
    mdtc = MDTC(3, 4, 23, 64, 5, True, 2)
    print(mdtc)

    num_params = sum(p.numel() for p in mdtc.parameters())
    print('the number of model params: {}'.format(num_params))
    x = torch.zeros(128, 200, 23)  # batch-size * time * dim
    y = mdtc(x)  # batch-size * time * dim
    print('input shape: {}'.format(x.shape))
    print('output shape: {}'.format(y.shape))
