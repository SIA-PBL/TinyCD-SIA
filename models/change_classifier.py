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
                UpMask(64, 56, 64),
                UpMask(128, 64, 64),
                UpMask(256, 64, 32),
            ]
        )

        # Final classification layer:
        self._classify = PixelwiseLinear([32, 16, 8], [16, 8, 1], Sigmoid())

    def forward(self, ref: Tensor, test: Tensor) -> Tensor:
        feat_refs, feat_tests, feat_mixed = self._encode(ref, test)
        for num, enc_ref, enc_test in enumerate(zip(feat_refs, feat_tests)):
            feat_mixed.append(self._mixing_mask[num](enc_ref, enc_test))

        latents = self._decode(feat_mixed)
        auxiliary_ref = self._segment(feat_refs)
        auxiliary_test = self._segment(feat_tests)
        print(auxiliary_ref)
        return self._classify(latents)

    def _encode(self, ref, test) -> List[Tensor]:
        '''
        내부 Mixing Mask 연산을 제거하고 Backbone에 의해 인코딩된
        피쳐 리스트만 추출하도록 수정
        Input: ref, test
        Output: ref, test에 _first_mix 적용한 값 들어있는 리스트 (features)
                레이어 별 encoded_ref 리스트 (feat_refs) 
                레이어 별 encoded_test 리스트 (feat_tests)
        '''
        feat_mixed = [self._first_mix(ref, test)]
        feat_refs = []
        feat_tests = []
        for layer in self._backbone:
            ref, test = layer(ref), layer(test)
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
각 레이어별로 추출된 enc_ref와 enc_test의 리스트인 feat_refs와 feat_tests를 입력 받아
concat 후 SegFormer의 decoder를 통과시켜 Segmentation을 구하는 함수
'''

def _segment(self, out_channels: int, features: List[Tensor]) -> Tensor:
    segformer = SegFormerBranch(out_channels=out_channels)
    sem_seg = segformer(features)
    return sem_seg

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
