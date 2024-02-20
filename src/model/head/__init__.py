from .simple import SimpleHead


def get_head(
        head_config, 
        backbone_out_feats
    ):
    head_type = head_config.type

    if head_type == 'simple':
        return SimpleHead(
            backbone_out_feats=backbone_out_feats,
            embeddings_size=head_config.embeddings_size,
            dropout_p=head_config.dropout_p)
    