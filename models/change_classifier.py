from typing import List
import torchvision
from models.layers import MixingMaskAttentionBlock, PixelwiseLinear, UpMask, MixingBlock, SegFormerBranch
from torch import Tensor
from torch.nn import Module, ModuleList, Sigmoid


class ChangeClassifier(Module):
    def __init__(
        self,
        bkbn_name="efficientnet_b4",
        pretrained=True,
        output_layer_bkbn="3",
        freeze_backbone=False,
    ):
        super().__init__()

        # Load the pretrained backbone according to parameters:
        self._backbone = _get_backbone(
            bkbn_name, pretrained, output_layer_bkbn, freeze_backbone
        )

        # Initialize mixing blocks:
        self._first_mix = MixingMaskAttentionBlock(
            6, 3, [3, 10, 5], [10, 5, 1])
        self._mixing_mask = ModuleList(
            [
                MixingMaskAttentionBlock(48, 24, [24, 12, 6], [12, 6, 1]),
                MixingMaskAttentionBlock(64, 32, [32, 16, 8], [16, 8, 1]),
                MixingBlock(112, 56),
            ]
        )

        # Initialize Upsampling blocks:
        self._up = ModuleList(
            [
                UpMask(2, 56, 64),
                UpMask(2, 64, 64),
                UpMask(2, 64, 32),
            ]
        )

        # Final classification layer:
        self._classify = PixelwiseLinear([32, 16, 8], [16, 8, 1], Sigmoid())
        # segmentation layers
        self._segment = SegFormerBranch([24, 32, 56], 56, [2, 4, 8], 1)

    def forward(self, ref: Tensor, test: Tensor) -> Tensor:
        feat_refs, feat_tests, feat_mixed = self._encode(ref, test)
        feat_reverse_refs, feat_reverse_tests, feat_reverse_mixed = self._encode(test, ref)
        for num, (enc_ref, enc_test) in enumerate(list(zip(feat_refs, feat_tests))):
            feat_mixed.append(self._mixing_mask[num](enc_ref, enc_test))
        for num, (enc_reverse_ref, enc_reverse_test) in enumerate(list(zip(feat_reverse_refs, feat_reverse_tests))):
            feat_reverse_mixed.append(self._mixing_mask[num](enc_reverse_ref, enc_reverse_test))

        latents = self._decode(feat_mixed)
        reverse_latents = self._decode(feat_reverse_mixed)
        auxiliary_ref = self._segment(feat_refs)
        auxiliary_test = self._segment(feat_tests)
        #print(auxiliary_ref)
        return self._classify(latents), self._classify(reverse_latents), auxiliary_ref, auxiliary_test

    def _encode(self, ref, test) -> List[Tensor]:
        '''
        ?????? Mixing Mask ????????? ???????????? Backbone??? ?????? ????????????
        ?????? ???????????? ??????????????? ??????
        Input: ref, test
        Output: ref, test??? _first_mix ????????? ??? ???????????? ????????? (features)
                ????????? ??? encoded_ref ????????? (feat_refs) 
                ????????? ??? encoded_test ????????? (feat_tests)
        '''
        feat_mixed = [self._first_mix(ref, test)]
        feat_refs = []
        feat_tests = []
        for num, layer in enumerate(self._backbone):
            ref, test = layer(ref), layer(test)
            if num != 0: # ignore raw image tensor
                feat_refs.append(ref)
                feat_tests.append(test)
        return feat_refs, feat_tests, feat_mixed

    def _decode(self, features) -> Tensor:
        upping = features[-1]
        for i, j in enumerate(range(-2, -5, -1)):
            upping = self._up[i](upping, features[j])
        return upping

# TODO
'''
??? ??????????????? ????????? enc_ref??? enc_test??? ???????????? feat_refs??? feat_tests??? ?????? ??????
concat ??? SegFormer??? decoder??? ???????????? Segmentation??? ????????? ??????
'''
'''
def _segment(self, width, out_channels: int, scale_factor: int, features: List[Tensor]) -> Tensor:
    segformer = SegFormerBranch(width=width, decoder_channels=out_channels, scale_factors=scale_factor)
    sem_seg = segformer(features)
    return sem_seg
'''

def _get_backbone(
    bkbn_name, pretrained, output_layer_bkbn, freeze_backbone
) -> ModuleList:
    # The whole model:
    entire_model = getattr(torchvision.models, bkbn_name)(
        pretrained=pretrained
    ).features

    # Slicing it:
    derived_model = ModuleList([])
    for name, layer in entire_model.named_children():
        derived_model.append(layer)
        if name == output_layer_bkbn:
            break

    # Freezing the backbone weights:
    if freeze_backbone:
        for param in derived_model.parameters():
            param.requires_grad = False
    return derived_model
