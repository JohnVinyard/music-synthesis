from torch.nn.init import xavier_normal_, calculate_gain


def basic_init(name, weight):
    raise Exception('Do not use this until biases can be ignored')
    if weight.data.dim() > 2:
        weight.data.normal_(0, 0.02)


def generator_init(name, weight):
    raise Exception('Do not use this until biases can be ignored')
    if weight.data.dim() > 2:
        if 'to_frames' in name:
            xavier_normal_(weight.data, 1)
        else:
            xavier_normal_(
                weight.data, calculate_gain('leaky_relu', 0.2))


def discriminator_init(name, weight):
    raise Exception('Do not use this until biases can be ignored')
    if weight.data.dim() > 2:
        if 'judge' in name:
            xavier_normal_(weight.data, 1)
        else:
            xavier_normal_(
                weight.data, calculate_gain('leaky_relu', 0.2))
