import numpy as np


def cal_map(fb_seq_list):
    aps = []
    for fb_seq in fb_seq_list:
        ap = 0.0
        pos = 0.0
        view = 0.0
        for r_index, (click, bounce) in enumerate(fb_seq):
            view += 1.0
            if click > 0:
                pos += 1.0
                ap += pos / (r_index + 1.0)
            if bounce > 0:
                break
        aps.append(ap / view)

    return np.mean(aps)


def cal_click_num(fb_seq_list):
    click_nums = []
    for fb_seq in fb_seq_list:
        click_num = 0.0
        for r_index, (click, bounce) in enumerate(fb_seq):
            click_num += click
            if bounce == 1:
                break
        click_nums.append(click_num)

    return np.mean(click_nums)


def cal_view_deep(fb_seq_list):
    click_nums = []
    for fb_seq in fb_seq_list:
        click_num = 0.0
        for r_index, (click, bounce) in enumerate(fb_seq):
            click_num += 1.0
            if bounce == 1:
                break
        click_nums.append(click_num)

    return np.mean(click_nums)


def cal_expection(ctr_list, pbr_list, features_mask):
    batch_size = len(ctr_list)
    rewards = []
    for b_index in range(batch_size):
        reward = 0.0
        keep = 1.0
        for index in range(int(features_mask[b_index])):
            reward += keep * ctr_list[b_index][index]
            keep *= (1.0 - pbr_list[b_index][index])
        rewards.append(reward)
    return rewards


def cal_view_depth_expection(pbr_list, features_mask):
    batch_size = len(pbr_list)
    rewards = []
    for b_index in range(batch_size):
        reward = 0.0
        keep = 1.0
        for index in range(int(features_mask[b_index])):
            reward += keep
            keep *= (1.0 - pbr_list[b_index][index])
        rewards.append(reward)
    return rewards


def cal_map_expection(ctr_list, pbr_list, features_mask):
    batch_size = len(ctr_list)
    aps = []
    for b_index in range(batch_size):
        keep = 1.0
        ap = 0.0
        pos = 0.0
        view = 0.0
        for index in range(int(features_mask[b_index])):
            view += 1.0
            pos += ctr_list[b_index][index]
            ap += keep * pos / (index + 1.0)
            keep *= (1.0 - pbr_list[b_index][index])
        aps.append(ap / view)
    return aps


def cal_map_static(selected_click, selected_index, features_mask):
    batch_size = len(selected_click)
    aps = []
    for b_index in range(batch_size):
        ap = 0.0
        pos = 0.0
        view = 0.0
        for index in range(int(features_mask[b_index])):
            s_index = selected_index[b_index][index]
            view += 1.0
            pos += float(selected_click[b_index][s_index])
            ap += pos / (index + 1.0)
        aps.append(ap / view)
    return aps
