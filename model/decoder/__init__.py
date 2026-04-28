# __init__.py：解码器调度中心
# 这是老规矩了，它根据你的配置文件（args.decoder_type）来选择到底用哪种解码器。
# 主要有三种选择：rnn_decoder（序列生成，你现在用的就是这个）、transformer（大语言模型常用的生成方式）和 tree_decoder（生成树状结构的公式）。

from config import decoder_list
from .rnn_decoder import DecoderRNN
from .tree_decoder import TreeDecoder
from .transformer import TransformerDecoder

def get_decoder(params, *args):
         
    if not params.decoder_type in decoder_list:
        raise NotImplementedError(
            "Unsupported Classifier: {}".format(params.decoder_type))

    if params.decoder_type == "transformer":
        decoder = TransformerDecoder(params, *args)
    elif params.decoder_type == "rnn_decoder":
        decoder = DecoderRNN(params, *args)
    elif params.decoder_type == "tree_decoder":
        decoder = TreeDecoder(params, *args)
    else:
        raise NotImplementedError("Unsupported Decoder: {}".format(params.decoder_type))
             
    return decoder


