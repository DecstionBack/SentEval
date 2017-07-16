"""
Use transformer to generate
conditional encoding
"""

from tensor2tensor.models import transformer
from tensor2tensor.utils import registry

if __name__ == '__main__':
    hparams = transformer.transformer_base()
    # model = transformer.Transformer(hparams=hparams, mode=)