from .search import create_model


# NOTE: Disign Philosophy

# Consistency makes life easier.

# All models are designed to have pretrained weights if possible.
# Users should re-initialize the weights if they want to train from scratch.

# All models are feature extractors, no task-specific heads are included, some models who orignally have heads will
# be re-wrapped to remove the heads.
# Users should add task-specific heads to the models according to pipeline requirements.

# All feature extractors are treated as a black box, no guarantee on its attributes execpt default forward method.
# Users should not access the attributes of the feature extractors directly. instead, design a model
# wrapper to use the feature extractor as a component.

# All submodule implementations should be independent to codes except themself, making them
# easy to be copied and pasted to other projects.
