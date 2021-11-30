import torch.nn as nn
import torch as t
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm


def norm_ip(img, min, max):
    temp = t.clamp(img, min=min, max=max)
    temp = (temp + -min) / (max - min + 1e-5)
    return temp


def cond_is_fid(f, replay_buffer, args, device, ratio=0.1, eval='all'):
    n_it = replay_buffer.size(0) // 100
    all_y = []
    probs = []
    with t.no_grad():
        for i in tqdm(range(n_it)):
            x = replay_buffer[i * 100: (i + 1) * 100].to(device)
            logits = f.classify(x)
            if 'dis' in args.model:
                logits = logits[0][0]
            y = logits.max(1)[1]
            prob = nn.Softmax(dim=1)(logits).max(1)[0]
            all_y.append(y)
            probs.append(prob)

    all_y = t.cat(all_y, 0)
    probs = t.cat(probs, 0)
    each_class = [replay_buffer[all_y == l] for l in range(args.n_classes)]
    each_class_probs = [probs[all_y == l] for l in range(args.n_classes)]
    print([len(c) for c in each_class])

    new_buffer = []
    for c in range(args.n_classes):
        each_probs = each_class_probs[c]
        # print("%d" % len(each_probs))
        if ratio <= 1:
            topk = int(len(each_probs) * ratio)
        else:
            topk = int(ratio)
        topk = min(topk, len(each_probs))
        topks = t.topk(each_probs, topk)
        index_list = topks[1]
        images = each_class[c][index_list]
        new_buffer.append(images)

    replay_buffer = t.cat(new_buffer, 0)
    from Task.eval_buffer import eval_is_fid
    inc_score, std, fid = eval_is_fid(replay_buffer, args, eval=eval)
    if eval in ['is', 'all']:
        print("Inception score of {} with std of {}".format(inc_score, std))
    if eval in ['fid', 'all']:
        print("FID of score {}".format(fid))
    return inc_score, std, fid


def eval_is_fid(replay_buffer, args, eval='all'):
    from Task.inception import get_inception_score
    from Task.fid import get_fid_score
    if isinstance(replay_buffer, list):
        images = replay_buffer[0]
    elif isinstance(replay_buffer, tuple):
        images = replay_buffer[0]
    else:
        images = replay_buffer

    feed_imgs = []
    for i, img in enumerate(images):
        n_img = norm_ip(img, -1, 1)
        new_img = n_img.cpu().numpy().transpose(1, 2, 0) * 255
        feed_imgs.append(new_img)

    feed_imgs = np.stack(feed_imgs)

    if 'cifar100' in args.dataset:
        from Task.data import Cifar100
        test_dataset = Cifar100(args, augment=False)
    elif 'cifar' in args.dataset:
        from Task.data import Cifar10
        test_dataset = Cifar10(args, full=True, noise=False)
    elif 'svhn' in args.dataset:
        from Task.data import Svhn
        test_dataset = Svhn(args, augment=False)
    else:
        assert False, 'dataset %s' % args.dataset
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True, drop_last=False)

    test_ims = []

    def rescale_im(im):
        return np.clip(im * 256, 0, 255).astype(np.uint8)

    for data_corrupt, data, label_gt in tqdm(test_dataloader):
        data = data.numpy()
        test_ims.extend(list(rescale_im(data)))

    # FID score
    # n = min(len(images), len(test_ims))
    fid = -1
    if eval in ['fid', 'all']:
        try:
            fid = get_fid_score(feed_imgs, test_ims)
            print("FID of score {}".format(fid))
        except:
            print("FID failed")
            fid = -1
    score, std = 0, 0
    if eval in ['is', 'all']:
        splits = max(1, len(feed_imgs) // 5000)
        score, std = get_inception_score(feed_imgs, splits=splits)
        print("Inception score of {} with std of {}".format(score, std))
    return score, std, fid
