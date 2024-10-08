#!bash

# for model in squeezenet mobilenet mobilenetv2 shufflenet shufflenetv2 vgg11 vgg13 vgg16 vgg19 densenet121 densenet161 densenet201 googlenet inceptionv3 inceptionv4 inceptionresnetv2 xception resnet18 resnet34 resnet50 resnet101 resnet152 preactresnet18 preactresnet34 preactresnet50 preactresnet101 preactresnet152 resnext50 resnext101 resnext152 attention56 attention92 seresnet18 seresnet34 seresnet50 seresnet101 seresnet152 nasnet wideresnet stochasticdepth18 stochasticdepth34 stochasticdepth50 stochasticdepth101
for model in inceptionresnetv2 xception resnet18 preactresnet18 attention56 seresnet18 nasnet wideresnet stochasticdepth18 resnet34 preactresnet34 seresnet34 stochasticdepth34 resnet50 preactresnet50 resnext50 seresnet50 stochasticdepth50 resnet101 preactresnet101 resnext101 attention92 seresnet101 stochasticdepth101 resnet152 preactresnet152 resnext152 seresnet152
do
    python train.py -net $model -gpu $1
done