

def weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        m.weight.data.normal_(0, 0.02)
        try:
            m.bias.data.fill_(0)
        except AttributeError:
            pass