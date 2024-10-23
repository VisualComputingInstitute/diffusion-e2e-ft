# @GonzaloMartinGarcia

from torch.nn import Conv2d, Parameter

# Function is based on an early commit of the Marigold GitHub repository.
def replace_unet_conv_in(unet, repeat=2):
    _weight = unet.conv_in.weight.clone() 
    _bias = unet.conv_in.bias.clone() 
    _weight = _weight.repeat((1, repeat, 1, 1))  
    # scale the activation magnitude
    _weight /= repeat
    _bias /= repeat
    # new conv_in channel
    _n_convin_out_channel = unet.conv_in.out_channels
    _new_conv_in = Conv2d(4*repeat, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    _new_conv_in.weight = Parameter(_weight)
    _new_conv_in.bias = Parameter(_bias)
    unet.conv_in = _new_conv_in
    # replace config
    unet.config['in_channels'] = 4*repeat
    return