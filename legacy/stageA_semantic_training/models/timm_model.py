from __init__ import *
from .unet_decoder import *
from args import *

def _check_reduction(reduction_factors):
    r_prev = 1
    for r in reduction_factors:
        if r / r_prev != 2:
            raise AssertionError('Reduction assumed to increase by 2: {}'.format(reduction_factors))
        r_prev = r

class Model(nn.Module):
    # See also TimmUniversalEncoder in segmentation_models_pytorch
    def __init__(self, pretrained=True, decoder_channels=(384, 192, 96, 64), out_indices=(0, 1, 2, 3)):
        super().__init__()
        name = args.default_encoder
        dropout = args.drop
        pretrained = pretrained

        self.encoder = timm.create_model(name, features_only=True, pretrained=pretrained, out_indices=out_indices)
        encoder_channels = self.encoder.feature_info.channels()

        _check_reduction(self.encoder.feature_info.reduction())

        print('Encoder channels:', name, encoder_channels)
        print('Decoder channels:', decoder_channels)

        assert len(encoder_channels) == len(decoder_channels)

        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            dropout=dropout,
        )

        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=1, activation=None, kernel_size=3,
        )

        initialize_decoder(self.decoder)


    def forward(self, x):
        # x: (batch_size, 3, 512, 512)
        # upsampled: (batch_size, 3, 1024, 1024)
        features = self.encoder(x)  # => list of 5 features

        decoder_output= self.decoder(features)
        y_pred = self.segmentation_head(decoder_output)
        # print('y_pred.shape:', y_pred.shape)

        return y_pred


if __name__ == "__main__":
    model = Model().cuda()
    input = torch.randn(1, 3, 256, 256).cuda()
    output = model(input)
    print(output.shape)