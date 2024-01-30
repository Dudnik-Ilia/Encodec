import torch
from model import EncodecModel

CHECKPOINT_OLD = "/home/woody/iwi1/iwi1010h/checkpoints/680242/bs3_cut36000_length0ep12_lr0.0003.pt"

def main(config):
    model = EncodecModel.my_encodec_model(checkpoint=CHECKPOINT_OLD)

    x = torch.rand(size=(1, 1, 48000))
    y = model.encode_decode_no_vq(x)
    return y


if __name__ == '__main__':
    main()