import torch.onnx
from torchvision.models.segmentation import deeplabv3_resnet50


def Convert_ONNX():

    # set the model to inference mode
    model.eval()

    # Let's create a dummy input tensor
    dummy_input = torch.randn(1, 5, 256, 256, requires_grad=True)

    # Export the model
    torch.onnx.export(model,
                      dummy_input,
                      "./models/test_torchgeo.onnx",
                      export_params=True,
                      do_constant_folding=True,
                      input_names=['input_img'],
                      output_names=['output_mask'])
    print('')
    print('Model has been converted to ONNX')


if __name__ == "__main__":

    # Load model
    weights = './models/test_torchgeo.pt'
    model = deeplabv3_resnet50(weights=None, num_classes=2)
    backbone = model.get_submodule('backbone')

    conv = torch.nn.modules.conv.Conv2d(
        in_channels=5,
        out_channels=64,
        kernel_size=(7, 7),
        stride=(2, 2),
        padding=(3, 3),
        bias=False
    )

    backbone.register_module('conv1', conv)
    model.load_state_dict(torch.load(weights))

    # Conversion to ONNX
    Convert_ONNX()
