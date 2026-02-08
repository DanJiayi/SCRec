# Re-use RPG tokenizer for RPGGatedSemantic (same codebook / item2tokens).
from genrec.models.RPG.tokenizer import RPGTokenizer


class RPGGatedSemanticTokenizer(RPGTokenizer):
    """Same as RPGTokenizer; semantic embeddings are loaded inside the model."""
    pass
