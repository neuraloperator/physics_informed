import torch
import torch.autograd as autograd
from train_utils.utils import save_checkpoint, zero_grad
from train_utils.losses import LpLoss
from .utils import cal_mixgrad
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None


class Baselinetrainer(object):
    def __init__(self, model,
                 device=torch.device('cpu'),
                 log=False, log_args=None):
        self.model = model.to(device)
        self.device = device
        self.log_init(log, log_args)

    def prepare_data(self, dataset):
        # collocation points
        self.col_xyzt = torch.from_numpy(dataset.col_xyzt).to(self.device).float()
        self.col_uvwp = torch.from_numpy(dataset.col_uvwp).to(self.device).float()
        # boundary points
        self.bd_xyzt = torch.from_numpy(dataset.bd_xyzt).to(self.device).float()
        self.bd_uvwp = torch.from_numpy(dataset.bd_uvwp).to(self.device).float()
        # initial condition
        self.ini_xyzt = torch.from_numpy(dataset.ini_xyzt).to(self.device).float()
        self.ini_uvwp = torch.from_numpy(dataset.ini_uvwp).to(self.device).float()

    def train_LBFGS(self, dataset,
                    optimizer):
        pass

    def train_adam(self,
                   optimizer,
                   alpha=100.0, beta=100.0,
                   iter_num=10,
                   path='beltrami', name='test.pt',
                   scheduler=None, re=1.0):
        self.model.train()
        self.col_xyzt.requires_grad = True
        mse = torch.nn.MSELoss()
        pbar = tqdm(range(iter_num), dynamic_ncols=True, smoothing=0.01)
        for e in pbar:
            optimizer.zero_grad()
            zero_grad(self.col_xyzt)

            pred_bd_uvwp = self.model(self.bd_xyzt)
            bd_loss = mse(pred_bd_uvwp[0:3], self.bd_uvwp[0:3])

            pred_ini_uvwp = self.model(self.ini_xyzt)
            ini_loss = mse(pred_ini_uvwp[0:3], self.ini_uvwp[0:3])

            pred_col_uvwp = self.model(self.col_xyzt)
            f_loss = self.loss_f(pred_col_uvwp, self.col_xyzt, re=re)
            gt_loss = mse(pred_col_uvwp, self.col_uvwp)
            total_loss = alpha * bd_loss + beta * ini_loss + f_loss + gt_loss * 100
            total_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            pbar.set_description(
                (
                    f'Total loss: {total_loss.item():.6f}, f loss: {f_loss.item():.7f} '
                    f'Boundary loss : {bd_loss.item():.7f}, initial loss: {ini_loss.item():.7f} '
                    f'Gt loss: {gt_loss.item():.6f}'
                )
            )
            if e % 500 == 0:
                u_err, v_err, w_err = self.eval_error()
                print(f'u error: {u_err}, v error: {v_err}, w error: {w_err}')
        save_checkpoint(path, name, self.model)

    def eval_error(self):
        lploss = LpLoss()
        self.model.eval()
        with torch.no_grad():
            pred_uvwp = self.model(self.col_xyzt)
            u_error = lploss(pred_uvwp[:, 0], self.col_uvwp[:, 0])
            v_error = lploss(pred_uvwp[:, 1], self.col_uvwp[:, 1])
            w_error = lploss(pred_uvwp[:, 2], self.col_uvwp[:, 2])
        return u_error.item(), v_error.item(), w_error.item()

    @staticmethod
    def log_init(log, log_args):
        if wandb and log:
            wandb.init(project=log_args['project'],
                       entity='hzzheng-pino',
                       config=log_args,
                       tags=['BelflowData'])

    @staticmethod
    def loss_f(uvwp, xyzt, re=1.0):
        '''
        Index table
        u: 0, v: 1, w: 2, p: 3
        x: 0, y: 1, z: 2, t: 3

        Args:
            uvwp: output of model - (u, v, w, p)
            xyzt: input of model - (x, y, z, t)
            re: Reynolds number

        Returns:
            residual of NS
        '''
        u_xyzt, u_xx, u_yy, u_zz = cal_mixgrad(uvwp[:, 0], xyzt)
        v_xyzt, v_xx, v_yy, v_zz = cal_mixgrad(uvwp[:, 1], xyzt)
        w_xyzt, w_xx, w_yy, w_zz = cal_mixgrad(uvwp[:, 2], xyzt)
        p_xyzt, = autograd.grad(outputs=[uvwp[:, 3].sum()], inputs=xyzt,
                                create_graph=True)

        evp4 = u_xyzt[:, 0] + v_xyzt[:, 1] + w_xyzt[:, 2]

        evp1 = u_xyzt[:, 3] + torch.sum(uvwp[:, :3] * u_xyzt[:, :3], dim=1) \
               + p_xyzt[:, 0] - (u_xx + u_yy + u_zz) / re
        evp2 = v_xyzt[:, 3] + torch.sum(uvwp[:, :3] * v_xyzt[:, :3], dim=1) \
               + p_xyzt[:, 1] - (v_xx + v_yy + v_zz) / re
        evp3 = w_xyzt[:, 3] + torch.sum(uvwp[:, :3] * w_xyzt[:, :3], dim=1) \
               + p_xyzt[:, 2] - (w_xx + w_yy + w_zz) / re

        return torch.mean(evp1 ** 2 + evp2 ** 2 + evp3 ** 2 + evp4 ** 2)
