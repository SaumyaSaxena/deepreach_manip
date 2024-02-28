import wandb
import torch
import inspect

from dynamics import dynamics 
from experiments import experiments
from utils import modules, dataio, losses

if __name__ == "__main__":
    run_path = 'iam-lab/deepreach/039rkjwx'

    api = wandb.Api()
    run = api.run(run_path)
    train_cfg = run.config

    ckpt_name = "checkpoints/epoch_9999_step_0_loss_2766.015.pth"
    ckpt_file = wandb.restore(ckpt_name, run_path=run_path,
                                root=run_path+'/checkpoints', replace=True)
    checkpoint = torch.load(ckpt_file.name)

    
    import ipdb; ipdb.set_trace()

    dynamics_class = getattr(dynamics, train_cfg.dynamics_class)
    dynamics = dynamics_class(**{argname: getattr(train_cfg, argname) for argname in inspect.signature(dynamics_class).parameters.keys() if argname != 'self'})

    dataset = dataio.ReachabilityDataset(
        dynamics=dynamics, numpoints=train_cfg.numpoints, 
        pretrain=train_cfg.pretrain, pretrain_iters=train_cfg.pretrain_iters, 
        tMin=train_cfg.tMin, tMax=train_cfg.tMax, 
        counter_start=train_cfg.counter_start, counter_end=train_cfg.counter_end, 
        num_src_samples=train_cfg.num_src_samples, num_target_samples=train_cfg.num_target_samples)

    model = modules.SingleBVPNet(in_features=dynamics.input_dim, out_features=1, type=train_cfg.model, mode=train_cfg.model_mode,
                                final_layer_factor=1., hidden_features=train_cfg.num_nl, num_hidden_layers=train_cfg.num_hl)
    model.cuda()
    model.load_state_dict(checkpoint)

    model.eval()
    model.requires_grad_(False)

    experiment_class = getattr(experiments, train_cfg.experiment_class)
    experiment = experiment_class(model=model, dataset=dataset, experiment_dir=experiment_dir)
    experiment.init_special(**{argname: getattr(train_cfg, argname) for argname in inspect.signature(experiment_class.init_special).parameters.keys() if argname != 'self'})

    