import tensorflow as tf 
import pandas as pd
import numpy as np
import sys

sys.path.append("..")

from vit.configs import base_config
from vit.layers import mha
from vit.models import ViTClassifierExtended
import gs
from pprint import pformat
from vit_keras import vit, utils
import tensorflow_io as tfio
#load model with weights

with tf.io.gfile.GFile('gs://vit_models/augreg/index.csv') as f:
  df = pd.read_csv(f)

#Best top-1 accuracy on ImageNet-1k
b16s = df.query(
    'ds=="i21k" & adapt_resolution==224 & adapt_ds=="imagenet2012" & name=="B/16"'
).sort_values("adapt_final_test", ascending=False)
b16s.head()

best_b16_i1k_checkpoint = str(b16s.iloc[0]["adapt_filename"])
b16s.iloc[0]["adapt_filename"], b16s.iloc[0]["adapt_final_test"]
filename = best_b16_i1k_checkpoint
#!gsutil cp {path} .
path = f"gs://vit_models/augreg/{filename}.npz"
local_path = path.split("//")[-1].split("/")[-1]
local_path
with open(local_path, "rb") as f:
    params_jax = np.load(f)
    params_jax = dict(zip(params_jax.keys(), params_jax.values()))

#load ViT model in TF
def load_model():
    num_classes = 1000
    config = base_config.get_config()
    with config.unlocked():
        config.num_classes = num_classes 

    config.to_dict()
    # add model with correct head

    # Make sure it works.
    inputs = tf.ones((1, 224, 224, 3))
    vit_b16_model = ViTClassifierExtended(config)
    vit_b16_model(tf.ones((1, 224, 224, 3)))[0].shape
    #outputs, attention_weights = vit_b16_model(inputs)
    #Copy the projection layer params
    # Projection.

    vit_b16_model.layers[0].layers[0].kernel.assign(
        tf.Variable(params_jax["embedding/kernel"])
    )
    vit_b16_model.layers[0].layers[0].bias.assign(tf.Variable(params_jax["embedding/bias"]))
    print(" ")

    np.testing.assert_allclose(
        vit_b16_model.layers[0].layers[0].kernel.numpy(), params_jax["embedding/kernel"]
    )
    np.testing.assert_allclose(
        vit_b16_model.layers[0].layers[0].bias.numpy(), params_jax["embedding/bias"]
    )

    #Copy the positional embeddings
    # Positional embedding.

    vit_b16_model.positional_embedding.assign(
        tf.Variable(params_jax["Transformer/posembed_input/pos_embedding"])
    )
    print(" ")

    np.testing.assert_allclose(
        vit_b16_model.positional_embedding.numpy(),
        params_jax["Transformer/posembed_input/pos_embedding"],
    )

    # Copy the cls_token
    # Cls token.

    vit_b16_model.cls_token.assign(tf.Variable(params_jax["cls"]))
    print(" ")
    np.testing.assert_allclose(vit_b16_model.cls_token.numpy(), params_jax["cls"])
    # Copy the final Layer Norm params
    # Final layer norm layer.
    vit_b16_model.layers[-2].gamma.assign(
        tf.Variable(params_jax["Transformer/encoder_norm/scale"])
    )
    vit_b16_model.layers[-2].beta.assign(
        tf.Variable(params_jax["Transformer/encoder_norm/bias"])
    )

    print(" ")

    np.testing.assert_allclose(
        vit_b16_model.layers[-2].gamma.numpy(), params_jax["Transformer/encoder_norm/scale"]
    )

    np.testing.assert_allclose(
        vit_b16_model.layers[-2].beta.numpy(), params_jax["Transformer/encoder_norm/bias"]
    )
    # Head layer.

    vit_b16_model.layers[-1].kernel.assign(tf.Variable(params_jax["head/kernel"]))
    vit_b16_model.layers[-1].bias.assign(tf.Variable(params_jax["head/bias"]))
    print(" ")
    np.testing.assert_allclose(
        vit_b16_model.layers[-1].kernel.numpy(), params_jax["head/kernel"]
    )
    np.testing.assert_allclose(
        vit_b16_model.layers[-1].bias.numpy(), params_jax["head/bias"]
    )

    def modify_attention_block(tf_component, jax_component, params_jax, config):
        tf_component.kernel.assign(
            tf.Variable(
                params_jax[f"{jax_component}/kernel"].reshape(config.projection_dim, -1)
            )
        )
        tf_component.bias.assign(
            tf.Variable(
                params_jax[f"{jax_component}/bias"].reshape(-1)
            )
        )
        return tf_component
    idx = 0
    for outer_layer in vit_b16_model.layers:
        if isinstance(outer_layer, tf.keras.Model) and outer_layer.name != "projection":
            tf_block = vit_b16_model.get_layer(outer_layer.name)
            jax_block_name = f"encoderblock_{idx}"

            # LayerNorm layers.
            layer_norm_idx = 0
            for layer in tf_block.layers:
                if isinstance(layer, tf.keras.layers.LayerNormalization):
                    layer_norm_jax_prefix = (
                        f"Transformer/{jax_block_name}/LayerNorm_{layer_norm_idx}"
                    )
                    layer.gamma.assign(
                        tf.Variable(params_jax[f"{layer_norm_jax_prefix}/scale"])
                    )
                    layer.beta.assign(
                        tf.Variable(params_jax[f"{layer_norm_jax_prefix}/bias"])
                    )
                    layer_norm_idx += 2

            # FFN layers.
            ffn_layer_idx = 0
            for layer in tf_block.layers:
                if isinstance(layer, tf.keras.layers.Dense):
                    dense_layer_jax_prefix = (
                        f"Transformer/{jax_block_name}/MlpBlock_3/Dense_{ffn_layer_idx}"
                    )
                    layer.kernel.assign(
                        tf.Variable(params_jax[f"{dense_layer_jax_prefix}/kernel"])
                    )
                    layer.bias.assign(
                        tf.Variable(params_jax[f"{dense_layer_jax_prefix}/bias"])
                    )
                    ffn_layer_idx += 1

            # Attention layer.
            for layer in tf_block.layers:
                attn_layer_jax_prefix = (
                    f"Transformer/{jax_block_name}/MultiHeadDotProductAttention_1"
                )
                if isinstance(layer, mha.TFViTAttention):
                    # Key
                    layer.self_attention.key = modify_attention_block(
                        layer.self_attention.key,
                        f"{attn_layer_jax_prefix}/key",
                        params_jax,
                        config,
                    )
                    # Query
                    layer.self_attention.query = modify_attention_block(
                        layer.self_attention.query,
                        f"{attn_layer_jax_prefix}/query",
                        params_jax,
                        config,
                    )
                    # Value
                    layer.self_attention.value = modify_attention_block(
                        layer.self_attention.value,
                        f"{attn_layer_jax_prefix}/value",
                        params_jax,
                        config,
                    )
                    # Final dense projection
                    layer.dense_output.dense.kernel.assign(
                        tf.Variable(
                            params_jax[f"{attn_layer_jax_prefix}/out/kernel"].reshape(
                                -1, config.projection_dim
                            )
                        )
                    )
                    layer.dense_output.dense.bias.assign(
                        tf.Variable(params_jax[f"{attn_layer_jax_prefix}/out/bias"])
                    )

            idx += 1

    idx = 0
    for outer_layer in vit_b16_model.layers:
        if isinstance(outer_layer, tf.keras.Model) and outer_layer.name != "projection":
            tf_block = vit_b16_model.get_layer(outer_layer.name)
            jax_block_name = f"encoderblock_{idx}"

            # Layer norm.
            layer_norm_idx = 0
            for layer in tf_block.layers:
                if isinstance(layer, tf.keras.layers.LayerNormalization):

                    layer_norm_jax_prefix = (
                        f"Transformer/{jax_block_name}/LayerNorm_{layer_norm_idx}"
                    )
                    np.testing.assert_allclose(
                        layer.gamma.numpy(), params_jax[f"{layer_norm_jax_prefix}/scale"]
                    )
                    np.testing.assert_allclose(
                        layer.beta.numpy(), params_jax[f"{layer_norm_jax_prefix}/bias"]
                    )
                    layer_norm_idx += 2

            # FFN layers.
            ffn_layer_idx = 0
            for layer in tf_block.layers:
                if isinstance(layer, tf.keras.layers.Dense):
                    dense_layer_jax_prefix = (
                        f"Transformer/{jax_block_name}/MlpBlock_3/Dense_{ffn_layer_idx}"
                    )
                    np.testing.assert_allclose(
                        layer.kernel.numpy(), params_jax[f"{dense_layer_jax_prefix}/kernel"]
                    )
                    np.testing.assert_allclose(
                        layer.bias.numpy(), params_jax[f"{dense_layer_jax_prefix}/bias"]
                    )
                    ffn_layer_idx += 1

            # Attention layers.
            for layer in tf_block.layers:
                attn_layer_jax_prefix = (
                    f"Transformer/{jax_block_name}/MultiHeadDotProductAttention_1"
                )
                if isinstance(layer, mha.TFViTAttention):

                    # Key
                    np.testing.assert_allclose(
                        layer.self_attention.key.kernel.numpy(),
                        params_jax[f"{attn_layer_jax_prefix}/key/kernel"].reshape(
                            config.projection_dim, -1
                        ),
                    )
                    np.testing.assert_allclose(
                        layer.self_attention.key.bias.numpy(),
                        params_jax[f"{attn_layer_jax_prefix}/key/bias"].reshape(-1),
                    )
                    # Query
                    np.testing.assert_allclose(
                        layer.self_attention.query.kernel.numpy(),
                        params_jax[f"{attn_layer_jax_prefix}/query/kernel"].reshape(
                            config.projection_dim, -1
                        ),
                    )
                    np.testing.assert_allclose(
                        layer.self_attention.query.bias.numpy(),
                        params_jax[f"{attn_layer_jax_prefix}/query/bias"].reshape(-1),
                    )
                    # Value
                    np.testing.assert_allclose(
                        layer.self_attention.value.kernel.numpy(),
                        params_jax[f"{attn_layer_jax_prefix}/value/kernel"].reshape(
                            config.projection_dim, -1
                        ),
                    )
                    np.testing.assert_allclose(
                        layer.self_attention.value.bias.numpy(),
                        params_jax[f"{attn_layer_jax_prefix}/value/bias"].reshape(-1),
                    )

                    # Final dense projection
                    np.testing.assert_allclose(
                        layer.dense_output.dense.kernel.numpy(),
                        params_jax[f"{attn_layer_jax_prefix}/out/kernel"].reshape(
                            -1, config.projection_dim
                        ),
                    )
                    np.testing.assert_allclose(
                        layer.dense_output.dense.bias.numpy(),
                        tf.Variable(params_jax[f"{attn_layer_jax_prefix}/out/bias"]),
                    )

            idx += 1
    return vit_b16_model