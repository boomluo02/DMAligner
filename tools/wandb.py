'''
Settings for Wandb and Sweep
'''
import torch
import wandb
from accelerate import Accelerator

def wandb_init(accelerator:Accelerator, cfg):
     # wandb init
     if accelerator.is_main_process:
        accelerator.init_trackers(project_name=cfg.env.project,
                                  config=cfg,
                                  init_kwargs={
                                      "entity": cfg.env.wandb_entity,
                                      "name": cfg.env.signature,
                                      "id": wandb.util.generate_id()
                                  })

def wandb_metric_init(*models):
    for model in models:
        wandb.watch(model, log='all', log_freq=10)

def wandb_log_criterion(metrics_dict, epoch, mode='train'):
    '''
    Log the criterion on wandb
    When num_gpus > 1, we need to average the criterion across all GPUs
    '''
    for key, value in metrics_dict.items():
        wandb.log({f'{mode}/{key}': value}, step = None if mode == 'test' else epoch+1)
    
def show_imgs_on_wandb(img_idx, **kwargs):
    '''
    Show the images on wandb
    img : OpenCV BGR, [C, H, W]
    '''
    epoch = kwargs.get('epoch')
    
    log_list = []
    for arg_k, arg_v in kwargs.items():
        if arg_k == 'epoch':
            continue
        caption = f'{img_idx}_{arg_k}.jpg'
        # tensor -> numpy
        if isinstance(arg_v, torch.Tensor):
            arg_v = arg_v.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255

        # BGR -> RGB
        # arg_v = cv2.cvtColor(arg_v, cv2.COLOR_BGR2RGB)
        log_list.append(wandb.Image(arg_v, caption=caption))

    wandb.log({f'{img_idx}': log_list}, step = None if epoch is None else epoch+1)

def wandb_log(tracker,
              img_1,
              img_2,
              pred_image, 
              gt_image,
              img_id, 
              step):
    log_list = []
    log_list.append(wandb.Image(img_1, caption=f"img1_{img_id}"))
    log_list.append(wandb.Image(img_2, caption=f"img2_{img_id}"))
    log_list.append(wandb.Image(pred_image, caption=f"pred_{img_id}"))
    log_list.append(wandb.Image(gt_image, caption=f"gt_{img_id}"))
    tracker.log(
        {   
            f"{img_id}": log_list
        },
        step = None if step is None else step
    )