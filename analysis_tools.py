from tensorflow import keras
import numpy as np
import tensorflow as tf
import vit_model_with_weights
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from custom_ssim import *
from matplotlib import pyplot as plt
from PIL import Image
import os
import pandas as pd

vit_b16_model = vit_model_with_weights.load_model()

#takes path as input, returns data ready to be preprocessed 
def get_stimulus(stimulus_path):
    #from PIL import Image
    image = Image.open(stimulus_path)
    return image

def get_stimuli(path_list):
    path_list.sort()
    stimuli = []
    for paths in path_list:
        for path in paths:
            stimulus = get_stimulus(path)
            stimulus = preprocess_image(stimulus)
            stimuli.append(stimulus)
    return(stimuli)

def get_rollout(stimulus):
    preprocessed_stimulus = preprocess_image(stimulus)
    att_rollout = get_att_rollout(stimulus, preprocessed_stimulus)
    return att_rollout

def calc_index(d1, d2):
    index = (d1-d2)/(d1+d2)
    index = round(index, 3)
    return index

def calc_mse(rollouts_array):
    mse1 = round(((rollouts_array[0] - rollouts_array[1])**2).mean(axis=None), 3)
    mse2 = round(((rollouts_array[2] - rollouts_array[3])**2).mean(axis=None), 3)
    return mse1, mse2

def get_rollouts(path_list):
    rollouts = []
    for paths in path_list:
        for path in paths:
            stimulus = get_stimulus(path)
            rollout = get_rollout(stimulus)
            rollouts.append(rollout)
    return(rollouts)

def create_metrics_dict(metrics):
    metrics_dict = {'lum_metric1': metrics[0], 'lum_metric2': metrics[1], 'struct_metric1': metrics[2], 'struct_metric2': metrics[3], 'mse1': metrics[4], 'mse2': metrics[5]}
    return metrics_dict

def create_idxs_dict(idxs):
    idxs_dict = {'lum_idx': idxs[0], 'struct_idx': idxs[1], 'mse_idx': idxs[2]}
    return idxs_dict

def get_metrics(rollouts):
        _, lum_metric1, struct_metric1 = get_ssim(rollouts[0], rollouts[1])
        _, lum_metric2, struct_metric2 = get_ssim(rollouts[2], rollouts[3])
        mse1, mse2 = calc_mse(rollouts)

        metrics = [lum_metric1, lum_metric2, struct_metric1, struct_metric2, mse1, mse2]
        metrics = create_metrics_dict(metrics)
        return metrics

def get_idxs(metrics):
    lum_idx = calc_index(metrics['lum_metric1'], metrics['lum_metric2'])
    struct_idx = calc_index(metrics['struct_metric1'], metrics['struct_metric2']) 
    mse_idx = calc_index(metrics['mse1'], metrics['mse2'])
    idxs = [lum_idx, struct_idx, mse_idx]
    idxs = create_idxs_dict(idxs)
    return idxs

def analyze(path_list):
    rollouts = get_rollouts(path_list)
    metrics = get_metrics(rollouts)
    idxs = get_idxs(metrics)
    return rollouts, metrics, idxs

def analyze_plain_photo(path_list):
    stimuli = get_stimuli(path_list)
    metrics = get_metrics(stimuli)
    idxs = get_idxs(metrics)
    return metrics, idxs

def make_lists(path):
    folders = os.listdir(path)
    folders.sort()
    final_list = []
    for folder in folders:
        imgs= os.listdir(path + folder)
        imgs.sort()
        for i in range(len(imgs)):
            imgs[i] = path + folder + '/' + imgs[i]

        imgs_cond_1 = []
        imgs_cond_1.append(imgs[0]) 
        imgs_cond_1.append(imgs[1]) 

        imgs_cond_2 = []
        imgs_cond_2.append(imgs[2])
        imgs_cond_2.append(imgs[3])

        combined_list = [imgs_cond_1, imgs_cond_2]
        final_list.append(combined_list)
    return final_list

def add_row_to_table(df_name, row_name, metrics, idxs):
    data = [metrics['lum_metric1'], metrics['lum_metric2'], idxs['lum_idx'], 
            metrics['struct_metric1'], metrics['struct_metric2'], idxs['struct_idx'], 
            metrics['mse1'], metrics['mse2'], idxs['mse_idx']]
    df_name[row_name] = data
    return

def plot_4_rollouts(rollouts, title):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(10, 8))
    #fig.suptitle(f"Predicted label: {predicted_label}.", fontsize=20)

    fig.suptitle(f"{title}", fontsize=20)
    _ = ax1.imshow(rollouts[0])
    _ = ax2.imshow(rollouts[1])
    _ = ax3.imshow(rollouts[2])
    _ = ax4.imshow(rollouts[3])

    ax1.set_title("rollout1", fontsize=16)
    ax2.set_title("rollout2", fontsize=16)
    ax3.set_title("rollout3", fontsize=16)
    ax4.set_title("rollout4", fontsize=16)

    ax1.axis("off")
    ax2.axis("off")
    ax3.axis("off")
    ax4.axis("off")

    fig.tight_layout()
    fig.subplots_adjust(top=1.2)
    #fig.show()
    return

def reshape_attentions(image, preprocessed=False):
    if preprocessed == False:
        _, attention_score_dict = vit_b16_model.predict(preprocess_image(image))
    else:
        _, attention_score_dict = vit_b16_model.predict(image)

    # Last transformer block.
    attention_scores = attention_score_dict["transformer_block_11_att"]

    patch_size = 16
    w_featmap = image.shape[2] // patch_size
    h_featmap = image.shape[1] // patch_size

    nh = attention_scores.shape[1]  # Number of attention heads.

    # Taking the representations from CLS token.
    attentions = attention_scores[0, :, 0, 1:].reshape(nh, -1)

    # Reshape the attention scores to resemble mini patches.
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = attentions.transpose((1, 2, 0))

    # Resize the attention patches to 224x224 (224: 14x16)
    attentions = tf.image.resize(attentions, size=(h_featmap*patch_size, w_featmap*patch_size))
    return attentions

import cv2
def get_att_rollout(image, preprocessed_image):
    #only takes un-processed image
    # preprocess if not preprocessed
    _, attention_score_dict = vit_b16_model.predict(preprocessed_image)

    num_cls_tokens = 1
    # Stack the individual attention matrices from individual transformer blocks.
    attn_mat = tf.stack([attention_score_dict[k] for k in attention_score_dict.keys()])
    attn_mat = tf.squeeze(attn_mat, axis=1)
    #print(attn_mat.shape)

    # Average the attention weights across all heads.
    attn_mat = tf.reduce_mean(attn_mat, axis=1)
    #print(attn_mat.shape)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_attn = tf.eye(attn_mat.shape[1])
    aug_attn_mat = attn_mat + residual_attn
    aug_attn_mat = aug_attn_mat / tf.reduce_sum(aug_attn_mat, axis=-1)[..., None]
    aug_attn_mat = aug_attn_mat.numpy()
    #print(aug_attn_mat.shape)

    # Recursively multiply the weight matrices
    joint_attentions = np.zeros(aug_attn_mat.shape)
    joint_attentions[0] = aug_attn_mat[0]

    for n in range(1, aug_attn_mat.shape[0]):
        joint_attentions[n] = np.matmul(aug_attn_mat[n], joint_attentions[n-1])

    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_attn_mat.shape[-1]))
    mask = v[0, num_cls_tokens:].reshape(grid_size, grid_size)
    mask = cv2.resize(mask / mask.max(), image.size)[..., np.newaxis]
    #result = (mask * image).astype("uint8")
    result = (mask * 255).astype("uint8")
    #print(result.shape)
    return result

#def add_row_to_plain_img_table(df_name, row_name, mses_up, mses_inv, ssims_up, ssims_inv):
    #data = [mses_up, mses_inv, ssims_up, ssims_inv]
    #df_name[row_name] = data
    #return

# Preprocessing steps required for attentional rollout
input_resolution = 224
model_type = 'vit'
patch_size = 16

crop_layer = keras.layers.CenterCrop(input_resolution, input_resolution)
norm_layer = keras.layers.Normalization(
    mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
    variance=[(0.229 * 255) ** 2, (0.224 * 255) ** 2, (0.225 * 255) ** 2],
)
rescale_layer = keras.layers.Rescaling(scale=1./127.5, offset=-1)


def preprocess_image(image, size=input_resolution):
    # turn the image into a numpy array and add batch dim
    image = np.array(image)
    image = tf.expand_dims(image, 0)
    
    # if model type is vit rescale the image to [-1, 1]
    if model_type == "vit":
        image = rescale_layer(image)

    # resize the image using bicubic interpolation
    resize_size = int((256 / 224) * size)
    image = tf.image.resize(
        image,
        (resize_size, resize_size),
        method="bicubic"
    )

    # crop the image
    image = crop_layer(image)

    #if model type is deit normalize the image
    if model_type != "vit":
        image = norm_layer(image)
    
    # return the image
    return image.numpy()

def analyze_plain_images(grouped_stimuli):
    metrics = []
    idxs = []
    for elem in grouped_stimuli:
        metric, idx = analyze_plain_photo(elem)
        metrics.append(metric)
        idxs.append(idx)
    return metrics, idxs

def test_plain_imgs(grouped_stimuli):
    stimuli = []
    for elem in grouped_stimuli:
        stimulus = get_stimuli(elem)
        stimuli.append(stimulus)
    mses_1 = []
    mses_2 = []
    ssims_1 = []
    ssims_2 = []

    for stimulus in stimuli:
        mse1, mse2 = calc_mse(stimulus)
        mses_1.append(mse1)
        mses_2.append(mse2)

        ssim_1, _, _ = get_ssim(stimulus[0], stimulus[1])
        ssim_2, _, _ = get_ssim(stimulus[2], stimulus[3])
        ssims_1.append(ssim_1)
        ssims_2.append(ssim_2)
    return mses_1, mses_2, ssims_1, ssims_2

def analyze_attention(grouped_stimuli):
    metrics = []
    idxs = []
    rollouts = []
    for elem in grouped_stimuli:
        rollout, metric, idx = analyze(elem)
        metrics.append(metric)
        idxs.append(idx)
        rollouts.append(rollout)

    return metrics, idxs, rollouts

def extract_idxs_from_dict(idxs_dict):
    lum_idxs = []
    struct_idxs = []
    mse_idxs = []
    for i in range(len(idxs_dict)):
        value = idxs_dict[i]
        lum_idxs.append(value['lum_idx'])
        struct_idxs.append(value['struct_idx'])
        mse_idxs.append(value['mse_idx'])

    return lum_idxs, struct_idxs, mse_idxs

def extract_metrics_from_dict(metrics_dict):
    lum_metric_1s = []
    lum_metric_2s = []
    struct_metric_1s = [] 
    struct_metric_2s = []
    mse_1s = []
    mse_2s = []
    for i in range(len(metrics_dict)):
        lum_metric_1s.append(metrics_dict[i]['lum_metric1']) 
        lum_metric_2s.append(metrics_dict[i]['lum_metric2']) 
        struct_metric_1s.append(metrics_dict[i]['struct_metric1']) 
        struct_metric_2s.append(metrics_dict[i]['struct_metric2'])
        mse_1s.append(metrics_dict[i]['mse1']) 
        mse_2s.append(metrics_dict[i]['mse2']) 
    return lum_metric_1s, lum_metric_2s, struct_metric_1s, struct_metric_2s, mse_1s, mse_2s

def make_excel_with_plain_img_data(plain_img_data, save_path):
    row_names = ['mse cond_1 vs cond_2 (d1)','mse cond_3 vs cond_4 (d2)', 
    'ssim cond_1 vs cond_2 (d1)', 'ssim cond_1 vs cond_2 (d2)', 
    'lum_d1', 'lum_d2', 'lum_idx', 
    'struct_d1', 'struct_d2', 'struct_idx',
    'mse_d1', 'mse_d2', 'mse_idx']
    plain_df = pd.DataFrame(row_names)

    lum_metrics_1, lum_metrics_2, struct_metrics_1, struct_metrics_2, mses_1, mses_2 = extract_metrics_from_dict(plain_img_data[4])
    lum_idxs, struct_idxs, mse_idxs = extract_idxs_from_dict(plain_img_data[5])

    for i in range(len(plain_img_data[0])):
        row_name = 'stimulus {}'.format(i)
        data = [plain_img_data[0][i], plain_img_data[1][i],
                round(float(plain_img_data[2][i]), 3), round(float(plain_img_data[3][i]), 3),
                lum_metrics_1[i], lum_metrics_2[i], lum_idxs[i],
                struct_metrics_1[i], struct_metrics_2[i], struct_idxs[i],
                mses_1[i], mses_2[i], mse_idxs[i]]
        plain_df[row_name] = data

    plain_df.to_excel(excel_writer=save_path + '/plain_image_data_sheet.xlsx')
    return

def test_everything(grouped_stimuli):
    plain_img_data = []
    plain_mses_1, plain_mses_2, plain_ssims_1, plain_ssims_2 = test_plain_imgs(grouped_stimuli)
    plain_metrics, plain_idxs = analyze_plain_images(grouped_stimuli)

    plain_img_data.append(plain_mses_1)
    plain_img_data.append(plain_mses_2)
    plain_img_data.append(plain_ssims_1)
    plain_img_data.append(plain_ssims_1)
    plain_img_data.append(plain_metrics)
    plain_img_data.append(plain_idxs)

    att_data = []
    att_metrics, att_idxs, att_rollouts = analyze_attention(grouped_stimuli)
    att_data.append(att_metrics)
    att_data.append(att_idxs) 
    att_data.append(att_rollouts)
    return plain_img_data, att_data

def make_excel_with_att_data(att_data, save_path):
    col_names = [['lum_metric1'], ['lum_metric2'], ['lum_idx'], 
            ['struct_metric1'], ['struct_metric2'], ['struct_idx'], 
            ['mse1'], ['mse2'], ['mse_idx']]
    att_df = pd.DataFrame(col_names)
    for i in range(len(att_data[0])):
        row_name = 'stimulus {}'.format(i)
        add_row_to_table(att_df, row_name, att_data[0][i], att_data[1][i])
    
    att_df.to_excel(excel_writer=save_path + '/att_data_sheet.xlsx')
    return

def plot_all_rollouts(rollouts, save_dir):
    rollout_dir = save_dir + '/rollouts/'
    os.mkdir(rollout_dir)
    os.chdir(rollout_dir)
    for i in range(len(rollouts)):
        plot_4_rollouts(rollouts[i], 'rollouts_for_stimulus{}'.format(i))
        plt.savefig('rollouts_for_stimulus{}'.format(i), bbox_inches='tight')
    return

def show_stimulus(abs_img_file_path):
    image = Image.open(abs_img_file_path)
    plt.imshow(image)
    return