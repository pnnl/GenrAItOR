from ..models import EmbedModel, LLMModel


def init_model(env):
    return LLMModel.build(env.model.name)


def init_embed_model(env):
    return EmbedModel.build(env.model.embed_name)
