import os
import logging
import argparse

from tqdm import tqdm
import torch
import torch.optim as optim
import numpy as np

from Task.eval_buffer import eval_is_fid
from utils import get_data, EMA, plot
from RecoveryLikelihood import RecoveryLikelihood
from models.wideresnet_te import Wide_ResNet as WResNet
from models.wrn_te_sn import Wide_ResNet as WResNetSN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(filename)s[%(lineno)d]: %(message)s",
    datefmt="%m-%d %H:%M:%S")
logger = logging.getLogger()
logger.setLevel(logging.INFO)
wlog = logger.info


def generate_from_scratch(diff_model, num_images, epoch, arg):
    noise = torch.randn(size=[25, 3, 32, 32]).to(arg.device)
    num_batch = int(np.ceil(num_images / 25))

    buffer = []
    for k in tqdm(range(num_batch)):
        x_neg, _ = diff_model.p_sample_progressive(noise)
        x_neg = x_neg[0].detach()
        buffer.append(x_neg)
        if k == 0 or k == 100:
            plot('{}/img_{}_{:>06d}.png'.format(arg.log_dir, epoch, k), x_neg)
    buffer = torch.cat(buffer)
    return buffer


class DRLTrainer:
    def __init__(self, arg):
        self.args = arg
        self.epoch_loss = 0
        self.diffusion = None
        self.diffusion_ema = None
        self.ema = None
        self.f_p = None

    def get_pred_by_freq(self, x, last=False):
        include_xpred_freq = max(1, self.args.num_diffusion_timesteps // 10)
        idx = torch.LongTensor(np.arange(self.args.num_diffusion_timesteps // include_xpred_freq + 1) * include_xpred_freq)
        if last:
            idx[-1] = idx[-1] - 1
        return x[idx]

    def train_a_batch(self, x, optimizer, epoch, i):
        t = torch.randint(size=[x.shape[0]], high=self.args.num_timesteps, device=device)

        x_pos, x_neg = self.diffusion.q_sample_pairs(x, t)
        x_neg, disp, is_accepted = self.diffusion.p_sample_langevin(x_neg, t)
        loss, _, f_p = self.diffusion.training_losses(x_pos, x_neg, t)
        self.epoch_loss += loss.item()
        self.f_p += f_p
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # self.ema.apply(self.diffusion)
        if i % 100 == 0:
            wlog('%d, [epoch:%d, iter:%d] Loss: %.03f' % (self.args.pid, epoch, i, self.epoch_loss / (i + 1)))
            f_ts = self.get_pred_by_freq(self.f_p, last=True)
            f_ts = ", ".join(["".join(str(np.around(aa.cpu().numpy(), 3))) for aa in f_ts])
            wlog('Positive Energy T ={:s}'.format(f_ts))
        return loss.item()

    def train(self):
        arg = self.args

        arg.n_classes = 10
        # datasets
        label_loader, train_loader, valid_loader, test_loader = get_data(arg)

        # model
        if arg.model == "wrnte":
            net = WResNet(arg.depth, arg.width, num_classes=arg.n_classes, norm=None)
            net_ema = WResNet(arg.depth, arg.width, num_classes=arg.n_classes, norm=None)
        else:
            net = WResNetSN(arg.depth, arg.width, num_classes=arg.n_classes, norm=None)
            net_ema = WResNetSN(arg.depth, arg.width, num_classes=arg.n_classes, norm=None)

        print('Model parameters: {:.2f}M'.format(sum(p.numel() for p in net.parameters()) / 1e6))
        if arg.resume is not None:
            pt = torch.load(args.resume)
            net.load_state_dict(pt)

        net = net.to(args.device)

        if args.lr > 0.01:
            optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)
        else:
            optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=[.9, .999], weight_decay=5e-4)

        self.diffusion = RecoveryLikelihood(net, arg)
        self.ema = EMA(mu=arg.ma_decay)
        # self.diffusion_ema = RecoveryLikelihood(net_ema, arg)
        # buffer = generate_from_scratch(self.diffusion, 500, epoch=-1, arg=arg)
        # inc_score, std, fid = eval_is_fid(buffer, arg, eval='all')
        # wlog("Inception score of {} with std of {}".format(inc_score, std))
        # wlog("FID of score {}".format(fid))

        best_acc = 0
        cur_iter = 0
        cur_lr = arg.lr
        for epoch in range(args.n_epochs):
            self.epoch_loss = 0
            self.f_p = 0
            if epoch in [120, 160, 200, 230]:
                cur_lr *= 0.2
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.2
                wlog("Learning rate decay %f" % cur_lr)

            net.train()
            for i, data in tqdm(enumerate(train_loader)):
                if cur_iter <= args.warmup_iters:
                    lr = args.lr * cur_iter / float(args.warmup_iters)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                x_p_d, _ = label_loader.__next__()
                x_p_d = x_p_d.to(device)

                loss = self.train_a_batch(x_p_d, optimizer, epoch, i)
                cur_iter += 1
                if abs(loss) > 1000:
                    print("diverge Epoch {} iter {}".format(epoch, i))
                    return

            metrics = {}
            i += 1  # in case of debugging
            if epoch % 5 == 0:
                buffer = generate_from_scratch(self.diffusion, 10000, epoch=epoch, arg=arg)
                inc_score, std, fid = eval_is_fid(buffer, arg, eval='all')
                wlog("Inception score of {} with std of {}".format(inc_score, std))
                wlog("FID of score {}".format(fid))
                metrics['Gen/IS'] = inc_score
                metrics['Gen/FID'] = fid
                metrics['Loss/EBM'] = self.epoch_loss / i
                metrics['Loss/T0'] = self.f_p[0] / i
                metrics['Loss/T2'] = self.f_p[2] / i
                metrics['Loss/T4'] = self.f_p[4] / i
                metrics['Loss/T6'] = self.f_p[5] / i

            torch.save(net.state_dict(), "./runs/DRL/%s/%s_last.pth" % (str(self.args.pid), str(arg.model)))
            if epoch % 10 == 0:
                torch.save(net.state_dict(), "./runs/DRL/%s/%s_%d.pth" % (str(self.args.pid), str(arg.model), epoch))


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Diffusion Recovery Likelihood")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "svhn", "cifar100"])
    parser.add_argument("--data_root", type=str, default="../data")
    # optimization
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=250)
    parser.add_argument("--warmup_iters", type=int, default=1000, help="number of iters to linearly increase learning rate, if -1 then no warmmup")
    parser.add_argument("--sigma", type=float, default=3e-2, help="stddev of gaussian noise to add to input, .03 works but .1 is more stable")
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    parser.add_argument("--width", type=int, default=10, help="WRN width parameter")
    parser.add_argument("--depth", type=int, default=28, help="WRN depth parameter")
    parser.add_argument("--model", type=str, default='wrnte', help='wrnte, wrntesn')
    parser.add_argument("--norm", type=str, default=None, choices=[None, "none", "batch", "instance", "layer", "act", 'td'], help="norm to add to weights, none works fine")

    parser.add_argument("--n_valid", type=int, default=0)
    parser.add_argument("--log_dir", type=str, default='./runs/DRL')
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--novis", action="store_true", help="")
    parser.add_argument("--debug", action="store_true", help="")
    parser.add_argument("--exp_name", type=str, default="DIS", help="exp name, for description")
    parser.add_argument("--gpu-id", type=str, default="0")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.seed = 1
    args.device = device

    # hyper-parameters of diffusion recovery likelihood
    args.img_sz = 32
    args.num_diffusion_timesteps = 6
    args.num_timesteps = 6
    args.opt = 'sgd'
    args.ma_decay = 0.999
    args.noise_scale = 1.0
    args.mcmc_num_steps = 30
    args.mcmc_step_size_b_square = 2e-4
    args.debug = False

    args.pid = os.getpid()

    logfile = os.path.join('runs', 'DRL', "%d_%s.log" % (args.pid, args.model))
    args.log_dir = 'runs/DRL/%d' % args.pid
    print("log dir is %s" % args.log_dir)
    os.makedirs(args.log_dir, exist_ok=True)
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d]: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    trainer = DRLTrainer(args)
    trainer.train()
