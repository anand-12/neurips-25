import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from torch.func import functional_call, jacrev
import os 
import json 

# --- Configuration for Fixed Hyperparameters ---
class Config:
    X_DIM = 784
    Y_DIM_ONEHOT = 10
    Y_DIM_OUT = 10
    R_DIM = 64
    Z_DIM = 32
    ENC_HIDDEN_DIM = 64
    DEC_HIDDEN_DIM = 64
    LR = 1e-4
    KL_WEIGHT = 0.01
    BATCH_SIZE_TRAIN = 32
    BATCH_SIZE_EVAL = 64
    Z_DIVERGENCE_THRESHOLD = 2.0 
    MAX_PROTOTYPES_PER_HEAD = 10
    MIN_SAMPLES_FOR_PROTOTYPE = 20
    NUM_Z_COLLECTIONS_FOR_PROTOTYPE = 5
    M_JACOBIAN_SAMPLES = 100
    NUM_CONTEXT_JACOBIAN = 100
    JACOBIAN_PROJ_REG = 1e-5
    OPM_PROJ_DIM = 100
    NUM_EVAL_BATCHES = 50
    FIXED_EVAL_CONTEXT = 30
    SPLIT_MNIST_CLASSES_PER_TASK = 2
    ROTATED_MNIST_MAX_ANGLE = 90.0
    DATA_ROOT = './data'
    SEED = 42
    NUM_TASKS = 10 
    EPOCHS_PER_TASK = 5
    EXPERIMENT_TYPE = 'PermutedMNIST'
    FIXED_HEAD_PER_TASK = False
    DEVICE = torch.device("cpu")

config = Config()
device = torch.device("cpu") 
task_data_train_global = []
task_data_test_global = []

def parse_args_and_setup_config():
    global config, device
    parser = argparse.ArgumentParser(description="Orthogonal Gradient Projection Neural Process (OGP-NP) for Continual Learning.")
    parser.add_argument('--experiment_type', type=str, default=config.EXPERIMENT_TYPE,
                        choices=['PermutedMNIST', 'SplitMNIST', 'RotatedMNIST'],
                        help='Type of MNIST experiment.')
    parser.add_argument('--num_tasks', type=int, default=config.NUM_TASKS, 
                        help='Number of tasks.')
    parser.add_argument('--epochs_per_task', type=int, default=config.EPOCHS_PER_TASK, help='Epochs per task.')
    parser.add_argument('--seed', type=int, default=config.SEED, help='Random seed.')
    parser.add_argument('--fixed_head_per_task', action='store_true', default=config.FIXED_HEAD_PER_TASK,
                        help='If set, spawns a new head for each task. Overrides dynamic head allocation.')
    parser.add_argument('--z_divergence_threshold', type=float, default=config.Z_DIVERGENCE_THRESHOLD,
                        help='Symmetric KL Divergence threshold to spawn new head (if not fixed_head_per_task).')
    cli_args = parser.parse_args()
    config.SEED = cli_args.seed
    config.NUM_TASKS = cli_args.num_tasks
    config.EPOCHS_PER_TASK = cli_args.epochs_per_task
    config.EXPERIMENT_TYPE = cli_args.experiment_type
    config.FIXED_HEAD_PER_TASK = cli_args.fixed_head_per_task
    config.Z_DIVERGENCE_THRESHOLD = cli_args.z_divergence_threshold
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.DEVICE = device 
    print(f"Global device set to: {device}")
    print(f"Running with configuration: {vars(config)}")

def prepare_task_data():
    global task_data_train_global, task_data_test_global, config
    task_data_train_global.clear(); task_data_test_global.clear()
    mnist_mean, mnist_std = (0.1307,), (0.3081,)
    normalize_transform = transforms.Normalize(mnist_mean, mnist_std)
    actual_num_cl_tasks = config.NUM_TASKS
    config.X_DIM = 28 * 28 
    if config.EXPERIMENT_TYPE == "PermutedMNIST":
        print("Loading Permuted MNIST...")
        config.Y_DIM_ONEHOT, config.Y_DIM_OUT = 10, 10
        full_transform = transforms.Compose([transforms.ToTensor(), normalize_transform])
        _mnist_train_full = torchvision.datasets.MNIST(root=config.DATA_ROOT, train=True, download=True, transform=full_transform)
        _mnist_test_full = torchvision.datasets.MNIST(root=config.DATA_ROOT, train=False, download=True, transform=full_transform)
        x_train_list = [_mnist_train_full[i][0].view(-1) for i in range(len(_mnist_train_full))]
        y_train_list = [_mnist_train_full[i][1] for i in range(len(_mnist_train_full))]
        x_train_base, y_train_base = torch.stack(x_train_list), torch.tensor(y_train_list)
        x_test_list = [_mnist_test_full[i][0].view(-1) for i in range(len(_mnist_test_full))]
        y_test_list = [_mnist_test_full[i][1] for i in range(len(_mnist_test_full))]
        x_test_base, y_test_base = torch.stack(x_test_list), torch.tensor(y_test_list)
        task_data_train_global.append((x_train_base, y_train_base))
        task_data_test_global.append((x_test_base, y_test_base))
        print(f"  PermutedMNIST Task 1 (Original): train {len(x_train_base)}, test {len(x_test_base)}")
        for i in range(1, actual_num_cl_tasks): 
            perm = np.random.permutation(config.X_DIM)
            task_data_train_global.append((x_train_base[:, perm], y_train_base))
            task_data_test_global.append((x_test_base[:, perm], y_test_base))
            print(f"  PermutedMNIST Task {i+1} (Permuted): train {len(x_train_base)}, test {len(x_test_base)}")
    elif config.EXPERIMENT_TYPE == "SplitMNIST":
        print(f"Loading Split MNIST (fixed {config.SPLIT_MNIST_CLASSES_PER_TASK} classes per task)...")
        config.Y_DIM_ONEHOT, config.Y_DIM_OUT = 10, 10
        if 10 % config.SPLIT_MNIST_CLASSES_PER_TASK != 0:
            raise ValueError("For SplitMNIST, 10 must be divisible by classes_per_split_fixed.")
        actual_num_cl_tasks = 10 // config.SPLIT_MNIST_CLASSES_PER_TASK
        print(f"  Derived number of CL tasks for SplitMNIST: {actual_num_cl_tasks}")
        mnist_train_raw = torchvision.datasets.MNIST(root=config.DATA_ROOT, train=True, download=True)
        mnist_test_raw = torchvision.datasets.MNIST(root=config.DATA_ROOT, train=False, download=True)
        x_train_01_chw = mnist_train_raw.data.float().unsqueeze(1)/255.0 
        y_train_all = mnist_train_raw.targets
        x_test_01_chw = mnist_test_raw.data.float().unsqueeze(1)/255.0
        y_test_all = mnist_test_raw.targets
        x_train_norm_chw = normalize_transform(x_train_01_chw)
        x_test_norm_chw = normalize_transform(x_test_01_chw)
        x_train_flat_all = x_train_norm_chw.view(-1, config.X_DIM)
        x_test_flat_all = x_test_norm_chw.view(-1, config.X_DIM)
        for i in range(actual_num_cl_tasks):
            task_labels = list(range(i * config.SPLIT_MNIST_CLASSES_PER_TASK, (i + 1) * config.SPLIT_MNIST_CLASSES_PER_TASK))
            train_mask = torch.isin(y_train_all, torch.tensor(task_labels))
            test_mask = torch.isin(y_test_all, torch.tensor(task_labels))
            task_data_train_global.append((x_train_flat_all[train_mask], y_train_all[train_mask]))
            task_data_test_global.append((x_test_flat_all[test_mask], y_test_all[test_mask]))
            print(f"  SplitMNIST Task {i+1}: labels {task_labels}, train {len(task_data_train_global[-1][0])}, test {len(task_data_test_global[-1][0])}")
    elif config.EXPERIMENT_TYPE == "RotatedMNIST":
        print(f"Loading Rotated MNIST (max angle {config.ROTATED_MNIST_MAX_ANGLE} for {actual_num_cl_tasks} tasks)...")
        config.Y_DIM_ONEHOT, config.Y_DIM_OUT = 10, 10
        mnist_train_raw = torchvision.datasets.MNIST(root=config.DATA_ROOT, train=True, download=True)
        mnist_test_raw = torchvision.datasets.MNIST(root=config.DATA_ROOT, train=False, download=True)
        x_train_01_chw = mnist_train_raw.data.float().unsqueeze(1)/255.0
        y_train_all = mnist_train_raw.targets
        x_test_01_chw = mnist_test_raw.data.float().unsqueeze(1)/255.0
        y_test_all = mnist_test_raw.targets
        current_max_angle = config.ROTATED_MNIST_MAX_ANGLE
        if actual_num_cl_tasks == 1:
            angles = [0.0] 
            if current_max_angle != 0.0: print(f"  INFO: RotatedMNIST with 1 task, using 0 rotation.")
        elif actual_num_cl_tasks > 1:
            angles = [i * (current_max_angle / (actual_num_cl_tasks - 1)) for i in range(actual_num_cl_tasks)]
        else: angles = []
        for task_idx, angle in enumerate(angles):
            print(f"  RotatedMNIST Task {task_idx+1}: rotation {angle:.2f} deg.")
            rot_train_chw = TF.rotate(x_train_01_chw, angle, interpolation=TF.InterpolationMode.BILINEAR)
            rot_test_chw = TF.rotate(x_test_01_chw, angle, interpolation=TF.InterpolationMode.BILINEAR)
            norm_rot_train_chw = normalize_transform(rot_train_chw)
            norm_rot_test_chw = normalize_transform(rot_test_chw)
            task_data_train_global.append((norm_rot_train_chw.view(-1, config.X_DIM), y_train_all))
            task_data_test_global.append((norm_rot_test_chw.view(-1, config.X_DIM), y_test_all))
    else: raise ValueError(f"Unsupported experiment_type: {config.EXPERIMENT_TYPE}")
    config.NUM_CL_TASKS_EFFECTIVE = actual_num_cl_tasks

class NPEncoder(nn.Module):
    def __init__(self,i,o,h,r):super(NPEncoder,self).__init__();self.fc=nn.Sequential(nn.Linear(i+o,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,r))
    def forward(self,x,y):return self.fc(torch.cat([x,y],dim=-1))
class LatentEncoder(nn.Module):
    def __init__(self,r,z):super(LatentEncoder,self).__init__();self.fc_mean=nn.Linear(r,z);self.fc_logvar=nn.Linear(r,z)
    def forward(self,r_agg):return self.fc_mean(r_agg),self.fc_logvar(r_agg)
class NPDecoder(nn.Module):
    def __init__(self,x,z,h,y_out):super(NPDecoder,self).__init__();self.fc=nn.Sequential(nn.Linear(z+x,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,y_out))
    def forward(self,z_s,x_t):z_r=z_s.unsqueeze(1).repeat(1,x_t.size(1),1);return self.fc(torch.cat([z_r,x_t],dim=-1))

def kl_divergence_diag_gaussians(m1,lv1,m2,lv2,eps=1e-8):
    v1,v2=torch.exp(lv1)+eps,torch.exp(lv2)+eps;log_r=lv2-lv1
    return 0.5*(torch.sum(v1/v2,dim=-1)+torch.sum((m2-m1).pow(2)/v2,dim=-1)-m1.size(-1)+torch.sum(log_r,dim=-1))
def symmetric_kl_divergence(m1,lv1,m2,lv2):return kl_divergence_diag_gaussians(m1,lv1,m2,lv2)+kl_divergence_diag_gaussians(m2,lv2,m1,lv1)

class OGPNP(nn.Module):
    def __init__(self,c):super(OGPNP,self).__init__();self.cfg=c;self.x_dim,self.y_dim_onehot,self.y_dim_out=c.X_DIM,c.Y_DIM_ONEHOT,c.Y_DIM_OUT;self.r_dim,self.z_dim=c.R_DIM,c.Z_DIM;self.enc_hidden_dim,self.dec_hidden_dim=c.ENC_HIDDEN_DIM,c.DEC_HIDDEN_DIM;self.xy_encoder=NPEncoder(self.x_dim,self.y_dim_onehot,self.enc_hidden_dim,self.r_dim);self.latent_encoder=LatentEncoder(self.r_dim,self.z_dim);self.phi_modules=nn.ModuleList([self.xy_encoder,self.latent_encoder]);self.decoders=nn.ModuleList();self.decoder_optimizers=[];self.task_prototypes_params=[];self.head_task_counts=[];self.task_to_head_map={};self.past_task_jacobians_stacked=None;self.device_internal=c.DEVICE
    def to(self,*a,**k):
        super().to(*a,**k)
        try:
            nd=next(self.parameters()).device
            if self.device_internal!=nd:self.device_internal=nd;[setattr(self,'task_prototypes_params',[[ (m.to(nd) if m.device!=nd else m,lv.to(nd) if lv.device!=nd else lv) for m,lv in pl if isinstance(m,torch.Tensor)and isinstance(lv,torch.Tensor)] for pl in self.task_prototypes_params])]
        except StopIteration:
            if a and isinstance(a[0],(str,torch.device)):self.device_internal=torch.device(a[0])
            elif k and 'device' in k:self.device_internal=torch.device(k['device'])
        return self
    def get_phi_parameters(self):return [p for m in self.phi_modules for p in m.parameters()]
    def aggregate(self,r_i):return torch.mean(r_i,dim=1) if r_i.size(1)>0 else torch.zeros(r_i.size(0),self.r_dim,device=r_i.device)
    def reparameterize(self,zm,zlv):std=torch.exp(0.5*zlv);eps=torch.randn_like(std);return zm+eps*std
    @torch.no_grad()
    def get_task_z_distribution_params(self,x_ctx,y_ctx_onehot):
        om=[self.xy_encoder.training,self.latent_encoder.training];self.xy_encoder.eval();self.latent_encoder.eval()
        if x_ctx.numel()==0 or x_ctx.size(1)==0:bs=x_ctx.size(0) if x_ctx.ndim>1 and x_ctx.size(0)>0 else 1;zm,zlv=torch.zeros(bs,self.z_dim,device=self.device_internal),torch.zeros(bs,self.z_dim,device=self.device_internal)
        else:r_i=self.xy_encoder(x_ctx,y_ctx_onehot);r_agg=self.aggregate(r_i);zm,zlv=self.latent_encoder(r_agg)
        self.xy_encoder.train(om[0]);self.latent_encoder.train(om[1]);return zm,zlv
    def _get_prototype_distribution(self,h_idx):
        if h_idx<len(self.task_prototypes_params) and self.task_prototypes_params[h_idx]:
            m=torch.stack([p[0] for p in self.task_prototypes_params[h_idx]]);lv=torch.stack([p[1] for p in self.task_prototypes_params[h_idx]])
            return m.mean(dim=0),lv.mean(dim=0)
        return None,None
    def _update_prototype_distribution(self,h_idx,nzm,nzlv):
        self.task_prototypes_params[h_idx].append((nzm.detach().clone().to(self.device_internal),nzlv.detach().clone().to(self.device_internal)))
        if len(self.task_prototypes_params[h_idx])>self.cfg.MAX_PROTOTYPES_PER_HEAD:self.task_prototypes_params[h_idx].pop(0)
    def _spawn_new_head(self,izm,izlv):
        nd=NPDecoder(self.x_dim,self.z_dim,self.dec_hidden_dim,self.y_dim_out).to(self.device_internal);self.decoders.append(nd);nhi=len(self.decoders)-1
        opt_d=optim.Adam(nd.parameters(),lr=self.cfg.LR);self.decoder_optimizers.append(opt_d)
        self.task_prototypes_params.append([]);self._update_prototype_distribution(nhi,izm,izlv);self.head_task_counts.append(1)
        print(f"Spawned new head {nhi} with LR {self.cfg.LR}");return nhi
    def decide_head_for_task(self,orig_task_idx,task_ctx_x,task_ctx_y_onehot):
        azm,azlv=None,None
        if task_ctx_x.size(1)>=self.cfg.MIN_SAMPLES_FOR_PROTOTYPE:
            czms,czlvs=[],[]
            for _ in range(self.cfg.NUM_Z_COLLECTIONS_FOR_PROTOTYPE):zm,zlv=self.get_task_z_distribution_params(task_ctx_x,task_ctx_y_onehot);czms.append(zm.squeeze(0));czlvs.append(zlv.squeeze(0))
            azm=torch.stack(czms).mean(dim=0) if czms else torch.zeros(self.z_dim,device=self.device_internal)
            azlv=torch.stack(czlvs).mean(dim=0) if czlvs else torch.zeros(self.z_dim,device=self.device_internal)
        else:print(f"Task {orig_task_idx+1}: Ctx size {task_ctx_x.size(1)} < min_samples {self.cfg.MIN_SAMPLES_FOR_PROTOTYPE}. Prior-like proto.");azm,azlv=torch.zeros(self.z_dim,device=self.device_internal),torch.zeros(self.z_dim,device=self.device_internal)
        if self.cfg.FIXED_HEAD_PER_TASK:nhi=len(self.decoders);print(f"Task {orig_task_idx+1}: Fixed head. Spawning head {nhi}.");ah=self._spawn_new_head(azm,azlv)
        else:
            ah,min_div=-1,float('inf')
            if not self.decoders:print(f"Task {orig_task_idx+1}: No heads. Spawning head 0.");ah=self._spawn_new_head(azm,azlv)
            else:
                for hi in range(len(self.decoders)):
                    pm,plv=self._get_prototype_distribution(hi)
                    if pm is not None and plv is not None:
                        div=symmetric_kl_divergence(azm,azlv,pm,plv)
                        if div<min_div:min_div,ah=div,hi
                if ah!=-1 and min_div<self.cfg.Z_DIVERGENCE_THRESHOLD:print(f"Task {orig_task_idx+1}: Reusing head {ah}. Div: {min_div:.3f} (Th: {self.cfg.Z_DIVERGENCE_THRESHOLD})");self._update_prototype_distribution(ah,azm,azlv);self.head_task_counts[ah]+=1
                else:reason=f"Min div {min_div:.3f}>=th" if ah!=-1 else "No suit head";ni=self._spawn_new_head(azm,azlv);print(f"Task {orig_task_idx+1}: {reason}. Spawning new dyn head {ni}.");ah=ni
        self.task_to_head_map[orig_task_idx]=ah;return ah
    
    # CORRECTED forward method signature and call
    def forward(self, x_context, y_context_onehot, x_target, head_idx): # Renamed for clarity
        if not (0 <= head_idx < len(self.decoders)):
             raise ValueError(f"Invalid head_idx {head_idx} for forward. Decoders available: {len(self.decoders)}")
        r_i_ctx = self.xy_encoder(x_context, y_context_onehot)
        r_agg = self.aggregate(r_i_ctx)
        z_mean, z_logvar = self.latent_encoder(r_agg)
        z_sample = self.reparameterize(z_mean, z_logvar)
        return self.decoders[head_idx](z_sample, x_target), z_mean, z_logvar

def ogp_np_loss(y_pred,y_trg_lbl,zm,zlv,kl_w):
    y_trg_lbl=y_trg_lbl.long();ce=F.cross_entropy(y_pred.view(-1,y_pred.size(-1)),y_trg_lbl.view(-1),reduction='mean')
    kl=torch.mean(-0.5*torch.sum(1+zlv-zm.pow(2)-zlv.exp(),dim=1));return ce+kl_w*kl,ce,kl

def get_context_target_split(x_seq_b,y_seq_lbl_b,y_dim_onehot_c,cfg):
    mb_s,tot_p,_=x_seq_b.shape
    if mb_s!=1:x_seq_b,y_seq_lbl_b=x_seq_b[0:1],y_seq_lbl_b[0:1]
    if tot_p==0:return x_seq_b,torch.empty_like(x_seq_b.expand(-1,-1,y_dim_onehot_c)),y_seq_lbl_b,x_seq_b,y_seq_lbl_b,False
    cr_min,cr_max,tr_min,tr_max=0.1,0.7,0.2,0.9
    min_c,max_c=max(1,int(tot_p*cr_min)),max(max(1,int(tot_p*cr_min)),int(tot_p*cr_max))
    n_ctx=0;
    if tot_p==1:n_ctx=0
    elif min_c>=tot_p:n_ctx=0
    else:n_ctx=np.random.randint(min_c,max_c+1) if min_c<=max_c else min_c
    n_ctx=min(n_ctx,tot_p-1 if tot_p>1 else 0);n_ctx=max(0,n_ctx)
    idxs=torch.randperm(tot_p,device=x_seq_b.device);ctx_idxs,rem_idxs=idxs[:n_ctx],idxs[n_ctx:]
    n_rem=rem_idxs.size(0);min_t,max_t=max(1,int(n_rem*tr_min)),max(max(1,int(n_rem*tr_min)),int(n_rem*tr_max))
    n_trg=0
    if n_rem>0:
        if min_t<max_t:n_trg=np.random.randint(min_t,max_t+1)
        elif min_t==max_t:n_trg=min_t
        n_trg=min(n_trg,n_rem)
    if n_trg==0 and n_rem>0:n_trg=n_rem
    if n_ctx==0 and n_trg==0 and tot_p>0:n_trg=tot_p;trg_idxs=idxs;ctx_idxs=torch.empty(0,dtype=torch.long,device=x_seq_b.device)
    else:trg_idxs=rem_idxs[:n_trg]
    x_ctx,y_ctx_lbl=x_seq_b[:,ctx_idxs,:],y_seq_lbl_b[:,ctx_idxs]
    x_trg,y_trg_lbl=x_seq_b[:,trg_idxs,:],y_seq_lbl_b[:,trg_idxs]
    y_ctx_onehot=F.one_hot(y_ctx_lbl.long(),num_classes=y_dim_onehot_c).float()
    valid=(x_ctx.size(1)>0 or x_trg.size(1)>0)
    if x_trg.size(1)==0 and tot_p>0:valid=False
    return x_ctx,y_ctx_onehot,y_ctx_lbl,x_trg,y_trg_lbl,valid

@torch.no_grad()
def get_ogp_data_for_jacobian(task_id,cfg):
    x_cpu,y_cpu=task_data_train_global[task_id];x_cpu,y_cpu=x_cpu.cpu(),y_cpu.cpu()
    tot_s=x_cpu.size(0)
    if tot_s==0:
        return None,None,None
    n_ctx,n_val=cfg.NUM_CONTEXT_JACOBIAN,cfg.M_JACOBIAN_SAMPLES
    if n_ctx+n_val>tot_s:
        if tot_s<=n_val:n_val,n_ctx=tot_s,0
        else:n_ctx=min(n_ctx,tot_s-n_val)
    if n_val==0:return None,None,None
    idxs_cpu=torch.randperm(tot_s,device=torch.device("cpu"));ctx_idx,val_idx=idxs_cpu[:n_ctx],idxs_cpu[n_ctx:n_ctx+n_val]
    ctx_x=x_cpu[ctx_idx].to(cfg.DEVICE).unsqueeze(0);ctx_y_lbl=y_cpu[ctx_idx].to(cfg.DEVICE).unsqueeze(0)
    ctx_y_oh=F.one_hot(ctx_y_lbl.long(),num_classes=cfg.Y_DIM_ONEHOT).float().to(cfg.DEVICE)
    trg_x=x_cpu[val_idx].to(cfg.DEVICE).unsqueeze(0)
    if trg_x.size(1)==0:return None,None,None
    if n_ctx==0:ctx_x,ctx_y_oh=torch.empty(1,0,cfg.X_DIM,device=cfg.DEVICE,dtype=x_cpu.dtype),torch.empty(1,0,cfg.Y_DIM_ONEHOT,device=cfg.DEVICE,dtype=torch.float)
    return ctx_x,ctx_y_oh,trg_x

def collect_jacobian_for_task_ogp_func(model,orig_task_id,ctx_x_i,ctx_y_oh_i,trg_x_val_i,cfg):
    md,ctx_x_i,ctx_y_oh_i,trg_x_val_i=model.device_internal,ctx_x_i.to(model.device_internal),ctx_y_oh_i.to(model.device_internal),trg_x_val_i.to(model.device_internal)
    phi_pt,phi_pn,xy_pnl,le_pnl=[],[],[],[]
    for n,p in sorted(model.xy_encoder.named_parameters(),key=lambda i:i[0]):phi_pt.append(p);phi_pn.append(f"xy.{n}");xy_pnl.append(n)
    n_pxy=len(xy_pnl);b_xy={n:b.to(md) for n,b in model.xy_encoder.named_buffers()}
    for n,p in sorted(model.latent_encoder.named_parameters(),key=lambda i:i[0]):phi_pt.append(p);phi_pn.append(f"le.{n}");le_pnl.append(n)
    n_ple=len(le_pnl);b_le={n:b.to(md) for n,b in model.latent_encoder.named_buffers()}
    phi_pt_jac=tuple(phi_pt)
    head_idx=model.task_to_head_map.get(orig_task_id)
    if head_idx is None:print(f"Err OGP: Task {orig_task_id} no head. Skip J.");return torch.empty(0,0,device=md)
    dec_mod_jac=model.decoders[head_idx]
    def comp_out_enc_phi_static(phi_rt_t,cx,cyoh,tx,xy_m,le_m,agg_f,fix_d_m,xy_pns,le_pns,xy_bs,le_bs,n_p_xy):
        p_xy_d={n:phi_rt_t[i] for i,n in enumerate(xy_pns)};p_le_d={n:phi_rt_t[i+n_p_xy] for i,n in enumerate(le_pns)}
        r_i=functional_call(xy_m,(p_xy_d,xy_bs),args=(cx,cyoh));r_agg=agg_f(r_i)
        zm,_=functional_call(le_m,(p_le_d,le_bs),args=(r_agg));yp=fix_d_m(zm,tx)
        return yp.flatten()
    model.xy_encoder.eval();model.latent_encoder.eval();dec_mod_jac.eval()
    J_i_t=jacrev(comp_out_enc_phi_static,argnums=0,has_aux=False)(phi_pt_jac,ctx_x_i,ctx_y_oh_i,trg_x_val_i,model.xy_encoder,model.latent_encoder,model.aggregate,dec_mod_jac,xy_pnl,le_pnl,b_xy,b_le,n_pxy)
    J_i_fl=[]
    if not J_i_t or len(J_i_t)!=len(phi_pt_jac):print(f"CRIT OGP Err: jacrev len mismatch. Exp {len(phi_pt_jac)}, got {len(J_i_t) if J_i_t else 'None'}.");return torch.empty(0,0,device=md)
    tot_out_dim=trg_x_val_i.size(0)*trg_x_val_i.size(1)*cfg.Y_DIM_OUT
    for i,J_p in enumerate(J_i_t):
        p_n=phi_pn[i]
        if J_p is None:p_numel=phi_pt_jac[i].numel();print(f"  Warn OGP: jacrev None for {p_n}. Zero fill ({tot_out_dim},{p_numel}).");J_i_fl.append(torch.zeros(tot_out_dim,p_numel,device=md))
        else:J_i_fl.append(J_p.reshape(J_p.shape[0],-1))
    if not J_i_fl:return torch.empty(0,0,device=md)
    return torch.cat(J_i_fl,dim=1).detach()

def project_gradients_ogp(model,cfg):
    if model.past_task_jacobians_stacked is None or model.past_task_jacobians_stacked.numel()==0:return
    J_old,ogp_d=model.past_task_jacobians_stacked,model.device_internal
    phi_p_w_g=[p for p in model.get_phi_parameters() if p.grad is not None]
    if not phi_p_w_g:return
    g_f=torch.cat([p.grad.flatten() for p in phi_p_w_g]).to(ogp_d)
    if J_old.shape[1]!=g_f.shape[0]:print(f"CRIT OGP Dim Err: J_old.col ({J_old.shape[1]})!=grad.dim ({g_f.shape[0]}). Skip proj.");return
    J_eff=J_old
    if J_old.size(0)>cfg.OPM_PROJ_DIM and cfg.OPM_PROJ_DIM>0:
        n_con,p_d=J_old.size(0),min(cfg.OPM_PROJ_DIM,J_old.size(0))
        rp=torch.randn(n_con,p_d,device=ogp_d);qr,_=torch.linalg.qr(rp);J_eff=qr.T@J_old
    A=J_eff@J_eff.T;A.diagonal().add_(cfg.JACOBIAN_PROJ_REG);B=J_eff@g_f
    try:L=torch.linalg.cholesky(A);x=torch.cholesky_solve(B.unsqueeze(-1),L).squeeze(-1);g_pf=g_f-J_eff.T@x
    except torch.linalg.LinAlgError:
        try:Ap=torch.linalg.pinv(A);x=Ap@B;g_pf=g_f-J_eff.T@x
        except torch.linalg.LinAlgError:g_pf=g_f;print("OGP: All solve fail. No proj.")
    off=0
    for p in phi_p_w_g:numel=p.numel();p.grad.data=g_pf[off:off+numel].view_as(p.grad);off+=numel

def train_ogp_np_task(model,orig_task_idx,assign_head_idx,opt_phi,cfg_obj):
    if not (0<=assign_head_idx<len(model.decoder_optimizers)):raise ValueError(f"Invalid head_idx {assign_head_idx} or optimizers not setup.")
    opt_dec_head=model.decoder_optimizers[assign_head_idx]
    x_train,y_train=task_data_train_global[orig_task_idx]
    if x_train.numel()==0:print(f"Task {orig_task_idx+1}: No train data.");return
    train_ds=TensorDataset(x_train,y_train);bs_tr=min(cfg_obj.BATCH_SIZE_TRAIN,len(train_ds))
    if bs_tr==0:return
    train_dl=DataLoader(train_ds,batch_size=bs_tr,shuffle=True,drop_last=(len(train_ds)>bs_tr))
    model.train()
    for i,dec_m in enumerate(model.decoders):dec_m.train(mode=(i==assign_head_idx));[p.requires_grad_(i==assign_head_idx) for p in dec_m.parameters()]
    [p.requires_grad_(True) for p in model.get_phi_parameters()]
    print(f"Train Orig Task {orig_task_idx+1} on Head {assign_head_idx} (Eps: {cfg_obj.EPOCHS_PER_TASK})...")
    for ep in range(cfg_obj.EPOCHS_PER_TASK):
        ep_l,ep_c,ep_tp,b_done=0.0,0,0,0
        for x_dlb,y_dlb_lbl in train_dl:
            x_dlb,y_dlb_lbl=x_dlb.to(cfg_obj.DEVICE),y_dlb_lbl.to(cfg_obj.DEVICE)
            if x_dlb.size(0)==0:continue
            n_pts_s=x_dlb.size(0);x_s_np,y_s_lbl_np=x_dlb.reshape(1,n_pts_s,-1),y_dlb_lbl.reshape(1,n_pts_s)
            x_c,y_c_oh,_,x_t,y_t_lbl,valid_s=get_context_target_split(x_s_np,y_s_lbl_np,model.y_dim_onehot,cfg_obj)
            if not valid_s:continue
            opt_phi.zero_grad(set_to_none=True);opt_dec_head.zero_grad(set_to_none=True)
            # Pass head_idx positionally
            y_p,zm,zlv=model(x_c,y_c_oh,x_t, assign_head_idx) 
            loss,_,_=ogp_np_loss(y_p,y_t_lbl,zm,zlv,cfg_obj.KL_WEIGHT)
            if torch.isnan(loss) or torch.isinf(loss):print(f"Warn: NaN/Inf loss ep {ep+1}. Skip.");continue
            loss.backward()
            if model.past_task_jacobians_stacked is not None and model.past_task_jacobians_stacked.numel()>0:project_gradients_ogp(model,cfg_obj)
            torch.nn.utils.clip_grad_norm_(model.get_phi_parameters(),1.0)
            torch.nn.utils.clip_grad_norm_(model.decoders[assign_head_idx].parameters(),1.0)
            opt_phi.step();opt_dec_head.step()
            ep_l+=loss.item();_,preds=torch.max(y_p.data,-1)
            ep_c+=(preds.squeeze()==y_t_lbl.squeeze()).sum().item();ep_tp+=y_t_lbl.numel();b_done+=1
        if b_done>0:print(f"  Ep {ep+1}: AvgLoss={ep_l/b_done:.4f}, Acc={100.*ep_c/ep_tp:.2f}%")
        else:print(f"  Ep {ep+1}: No valid batches.")

def evaluate_ogp_np_task(model,orig_task_idx_eval,cfg_obj):
    x_test,y_test=task_data_test_global[orig_task_idx_eval]
    if x_test.numel()==0:return 0.0
    head_idx=model.task_to_head_map.get(orig_task_idx_eval)
    if head_idx is None or not (0<=head_idx<len(model.decoders)):print(f"Eval Warn: No/Inv head for task {orig_task_idx_eval+1}. Acc=0.");return 0.0
    test_ds=TensorDataset(x_test,y_test);bs_ev=min(cfg_obj.BATCH_SIZE_EVAL,len(test_ds))
    if bs_ev==0:return 0.0
    test_dl=DataLoader(test_ds,batch_size=bs_ev,shuffle=False,drop_last=(len(test_ds)>bs_ev))
    model.eval();tot_c,tot_p=0,0
    with torch.no_grad():
        for i,(x_dlb,y_dlb_lbl) in enumerate(test_dl):
            if i>=cfg_obj.NUM_EVAL_BATCHES:break
            x_dlb,y_dlb_lbl=x_dlb.to(cfg_obj.DEVICE),y_dlb_lbl.to(cfg_obj.DEVICE)
            if x_dlb.size(0)==0:continue
            n_pts_s=x_dlb.size(0);x_s_np,y_s_lbl_np=x_dlb.reshape(1,n_pts_s,-1),y_dlb_lbl.reshape(1,n_pts_s)
            n_c=min(cfg_obj.FIXED_EVAL_CONTEXT,n_pts_s-1 if n_pts_s>1 else 0);n_c=max(0,n_c);n_t=n_pts_s-n_c
            if n_t<=0:continue
            ctx_idxs,trg_idxs=torch.arange(n_c,device=cfg_obj.DEVICE),torch.arange(n_c,n_pts_s,device=cfg_obj.DEVICE)
            x_c,y_c_lbl=x_s_np[:,ctx_idxs,:],y_s_lbl_np[:,ctx_idxs];x_t,y_t_lbl=x_s_np[:,trg_idxs,:],y_s_lbl_np[:,trg_idxs]
            y_c_oh=F.one_hot(y_c_lbl.long(),num_classes=model.y_dim_onehot).float()
            if n_c==0:x_c,y_c_oh=torch.empty(1,0,model.x_dim,device=cfg_obj.DEVICE,dtype=x_s_np.dtype),torch.empty(1,0,model.y_dim_onehot,device=cfg_obj.DEVICE,dtype=torch.float)
            if x_t.size(1)==0:continue
            # Pass head_idx positionally
            y_p,_,_=model(x_c,y_c_oh,x_t, head_idx) 
            _,preds=torch.max(y_p.data,-1);tot_c+=(preds.squeeze()==y_t_lbl.squeeze()).sum().item();tot_p+=y_t_lbl.numel()
    return 100.*tot_c/tot_p if tot_p>0 else 0.0

def run_continual_learning_ogp_np_experiment(cfg_obj):
    model = OGPNP(cfg_obj).to(cfg_obj.DEVICE)
    optimizer_phi = optim.Adam(model.get_phi_parameters(), lr=cfg_obj.LR)
    K = cfg_obj.NUM_CL_TASKS_EFFECTIVE
    all_tasks_accuracy_matrix = np.full((K, K), np.nan)
    avg_accs_stream, active_heads_stream = [], []
    forward_transfer_components = {} 

    for current_task_idx_loop in range(K):
        print(f"\n--- Proc. Orig. Task {current_task_idx_loop+1}/{K} ({cfg_obj.EXPERIMENT_TYPE}) ---")
        model.eval() 
        x_train_curr, y_train_curr = task_data_train_global[current_task_idx_loop]
        n_dec_pts = min(x_train_curr.size(0), cfg_obj.NUM_CONTEXT_JACOBIAN, 100)
        assigned_head = -1
        if n_dec_pts >= cfg_obj.MIN_SAMPLES_FOR_PROTOTYPE and x_train_curr.size(0) > 0:
            idx_dec = torch.randperm(x_train_curr.size(0))[:n_dec_pts]
            ctx_x_d = x_train_curr[idx_dec].to(model.device_internal).unsqueeze(0)
            ctx_y_d_lbl = y_train_curr[idx_dec].to(model.device_internal).unsqueeze(0)
            ctx_y_d_onehot = F.one_hot(ctx_y_d_lbl.long(),num_classes=model.y_dim_onehot).float()
            assigned_head = model.decide_head_for_task(current_task_idx_loop, ctx_x_d, ctx_y_d_onehot)
        else:
            print(f"Task {current_task_idx_loop+1}: Data too small for prototype. Fallback.")
            if cfg_obj.FIXED_HEAD_PER_TASK or not model.decoders:
                dummy_z_m, dummy_z_lv = torch.zeros(model.z_dim,device=model.device_internal), torch.zeros(model.z_dim,device=model.device_internal)
                assigned_head = model._spawn_new_head(dummy_z_m, dummy_z_lv)
            else: assigned_head = len(model.decoders)-1
            model.task_to_head_map[current_task_idx_loop] = assigned_head
            if assigned_head >= len(model.head_task_counts): model.head_task_counts.append(0) 
            model.head_task_counts[assigned_head] +=1
        active_heads_stream.append(len(model.decoders))
        print(f"Original Task {current_task_idx_loop+1} assigned to Head {assigned_head}.")
        train_ogp_np_task(model, current_task_idx_loop, assigned_head, optimizer_phi, cfg_obj)
        model.eval()
        print(f"Collecting J for orig. task {current_task_idx_loop+1} (head {assigned_head})...")
        ctx_x_j,ctx_y_j,val_x_j = get_ogp_data_for_jacobian(current_task_idx_loop,cfg_obj)
        if ctx_x_j is not None and val_x_j is not None and val_x_j.size(1)>0:
            J_i = collect_jacobian_for_task_ogp_func(model,current_task_idx_loop,ctx_x_j,ctx_y_j,val_x_j,cfg_obj)
            if J_i is not None and J_i.numel()>0:
                print(f"  J_task{current_task_idx_loop} (head {assigned_head}) shape: {J_i.shape}")
                phi_d = sum(p.numel() for p in model.get_phi_parameters())
                if J_i.shape[1]!=phi_d: print(f"  OGP J WARN: J dim {J_i.shape[1]}!=phi_dim {phi_d}")
                else:
                    if model.past_task_jacobians_stacked is None or model.past_task_jacobians_stacked.numel()==0:
                        model.past_task_jacobians_stacked = J_i
                    elif model.past_task_jacobians_stacked.shape[1]!=J_i.shape[1]:
                        print("OGP J WARN: Phi dim mismatch for J concat! Reset J_stacked."); model.past_task_jacobians_stacked=J_i
                    else: model.past_task_jacobians_stacked = torch.cat([model.past_task_jacobians_stacked,J_i],dim=0)
                if model.past_task_jacobians_stacked is not None: print(f"  Total J_stacked shape: {model.past_task_jacobians_stacked.shape}")
            else: print(f"  Failed to collect J for task {current_task_idx_loop+1}.")
        else: print(f"  Skipping J collection for task {current_task_idx_loop+1} (no OPM data).")
        current_stage_eval_accuracies = []
        print(f"\n--- Evaluating after training Original Task {current_task_idx_loop+1} ---")
        for eval_id in range(K):
            if eval_id <= current_task_idx_loop:
                acc = evaluate_ogp_np_task(model, eval_id, cfg_obj)
                all_tasks_accuracy_matrix[current_task_idx_loop, eval_id] = acc
                # Only add to current_stage_eval_accuracies if it's a task seen up to this point
                # This list is used for avg_acc_this_stage which is "avg acc over tasks seen so far"
                current_stage_eval_accuracies.append(acc) # This logic needs to be careful if eval_id can be > current_task_idx_loop for this list
                print(f"  Acc on Orig. Task {eval_id+1}: {acc:.2f}% (head {model.task_to_head_map.get(eval_id,'N/A')})")
            if eval_id == current_task_idx_loop + 1 and eval_id < K :
                acc_fwt_component = evaluate_ogp_np_task(model, eval_id, cfg_obj)
                forward_transfer_components[eval_id] = acc_fwt_component
                print(f"  Acc for FWT on future Task {eval_id+1} (before its training): {acc_fwt_component:.2f}%")
        
        # Corrected calculation for avg_acc_this_stage
        # It should average accuracies of tasks 0 to current_task_idx_loop, using the current row of the matrix
        avg_acc_this_stage = np.nanmean(all_tasks_accuracy_matrix[current_task_idx_loop, :current_task_idx_loop+1])
        avg_accs_stream.append(avg_acc_this_stage)
        print(f"  Avg acc (tasks 1 to {current_task_idx_loop+1}): {avg_acc_this_stage:.2f}% | Active Heads: {len(model.decoders)}")

    bwt = 0.0
    if K > 1:
        for j in range(K - 1):
            acc_kj = all_tasks_accuracy_matrix[K-1, j] 
            acc_jj = all_tasks_accuracy_matrix[j, j]   
            if not np.isnan(acc_kj) and not np.isnan(acc_jj): bwt += (acc_kj - acc_jj)
        bwt /= (K - 1)
    print(f"Backward Transfer (BWT): {bwt:.4f}")
    print(f"Forward Transfer Components (Acc on T_k after T_k-1 trained): {forward_transfer_components}")
    return all_tasks_accuracy_matrix, avg_accs_stream, active_heads_stream, bwt, forward_transfer_components

# --- Main Execution ---
if __name__ == '__main__':
    parse_args_and_setup_config() 
    prepare_task_data() 
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    mode_str_for_path = "FixedHead" if config.FIXED_HEAD_PER_TASK else "DynamicHeadSKL"
    results_dir_name = f"results_OGP-NP_{config.EXPERIMENT_TYPE}_{mode_str_for_path}_{config.NUM_CL_TASKS_EFFECTIVE}tasks_seed{config.SEED}_{timestamp}"
    os.makedirs(results_dir_name, exist_ok=True)
    print(f"Results will be saved in: {results_dir_name}")
    config_to_save = {k: str(v) if isinstance(v, torch.device) else v for k, v in vars(config).items()}
    with open(os.path.join(results_dir_name, 'config.json'), 'w') as f: json.dump(config_to_save, f, indent=4)
    print(f"Starting OGP-NP CL: {config.EXPERIMENT_TYPE}, {config.NUM_CL_TASKS_EFFECTIVE} eff tasks. Fixed heads: {config.FIXED_HEAD_PER_TASK}")
    start_time = time.time()
    all_task_accs_matrix, avg_accs, heads_log, bwt_final, fwt_comps = run_continual_learning_ogp_np_experiment(config)
    end_time = time.time()
    experiment_duration_minutes = (end_time - start_time) / 60
    print(f"\nExperiment finished in {experiment_duration_minutes:.2f} mins.")
    final_heads_count = heads_log[-1] if heads_log else 'N/A'
    if heads_log: print(f"Final #heads: {final_heads_count}")
    results_data = {
        "config_used": config_to_save,
        "all_tasks_accuracy_matrix": all_task_accs_matrix.tolist(),
        "average_accuracies_over_time": avg_accs,
        "active_heads_over_time": heads_log,
        "backward_transfer": bwt_final,
        "forward_transfer_components_A_k-1_k": fwt_comps,
        "final_average_accuracy": avg_accs[-1] if avg_accs else np.nan,
        "final_num_heads": final_heads_count,
        "experiment_duration_minutes": experiment_duration_minutes
    }
    results_file_path = os.path.join(results_dir_name, 'results.pt')
    torch.save(results_data, results_file_path)
    print(f"Numerical results saved to: {results_file_path}")

    # Plotting is commented out
    # n_eff = config.NUM_CL_TASKS_EFFECTIVE
    # fig,ax1 = plt.subplots(figsize=(13,8)); cmap=plt.cm.get_cmap('viridis',n_eff+2)
    # # ... (plotting code was here) ...
    # plot_filename = os.path.join(results_dir_name, f'plot_accuracies_heads.png')
    # # plt.savefig(plot_filename); print(f"Plot generation code is present but commented out. Plot would be: {plot_filename}"); 
    # # plt.show() 

    print(f"\n--- OGP-NP Final Accuracies Table ({config.EXPERIMENT_TYPE}, {mode_str_for_path}) ---")
    n_eff = config.NUM_CL_TASKS_EFFECTIVE # Use effective number of tasks
    hdr="Eval Task V|Train Stage->|"; [hdr:=hdr+f" {i+1:2d}    |" for i in range(n_eff)]; print(hdr); print("-" * len(hdr))
    for eval_id in range(n_eff):
        row=f"   Task {eval_id+1:2d}   | "; 
        for stage_idx in range(n_eff):
            acc_val = all_tasks_accuracy_matrix[stage_idx, eval_id]
            if stage_idx<eval_id: row+="   -    | "
            elif not np.isnan(acc_val): row+=f"{acc_val:7.2f} | "
            else: row+="  N/A   | "
        print(row)
    if avg_accs: print(f"\nFinal avg acc: {avg_accs[-1]:.2f}%")
    print(f"Final BWT: {bwt_final:.4f}")
    print(f"FWT Components (Acc on T_k+1 after T_k trained): {fwt_comps}")
