# --- START OF FILE utils/logger.py (đã chỉnh sửa cho bài toán mới) ---
import torch
import math
    
def log_to_screen(time_used, init_value, best_value, reward, costs_history, search_history,
                  batch_size, dataset_size, T):
    # reward
    print('\n', '-'*60)
    print('Avg total reward:'.center(35), '{:<10f} +- {:<10f}'.format(
            reward.mean(), torch.std(reward) / math.sqrt(batch_size)))
    # Trong bài toán mới, step reward và total reward là như nhau
    print('Avg step reward:'.center(35), '{:<10f} +- {:<10f}'.format(
            reward.mean(), torch.std(reward) / math.sqrt(batch_size)))
            
    # cost - best_value bây giờ là avg_reward trả về từ rollout
    print('-'*60)
    print('Avg final best value (reward):'.center(35), '{:<10f} +- {:<10f}'.format(
                best_value.mean(), 0.0)) # best_value chỉ là một số, không có std dev
    
    # LOẠI BỎ CÁC VÒNG LẶP GÂY LỖI
    # for per in range(500,T+1,500): ...
    
    # time
    print('-'*60)
    print('Avg used time per instance:'.center(35), '{:f}s'.format(
            time_used.mean() / dataset_size))
    print('-'*60, '\n')
    
def log_to_tb_val(tb_logger, time_used, init_value, best_value, reward, costs_history, search_history,
                  batch_size, val_size, dataset_size, T, epoch):
    
    if epoch is None: return # Không log nếu không phải trong training loop
    
    tb_logger.log_value('validation/avg_time_per_instance',  time_used.mean() / dataset_size, epoch)
    tb_logger.log_value('validation/avg_reward', reward.mean(), epoch)
    
    # best_value bây giờ là avg_reward
    tb_logger.log_value('validation/avg_best_value(reward)', best_value.mean(), epoch)

    # LOẠI BỎ CÁC LOG KHÔNG CÒN PHÙ HỢP
    # for per in range(20,100,20): ...

def log_to_tb_train(tb_logger, agent, Reward, ratios, bl_val_detached, total_cost, grad_norms, reward, entropy, approx_kl_divergence,
               reinforce_loss, baseline_loss, log_likelihood, initial_cost, mini_step):
    
    # Hàm này có thể giữ nguyên vì các giá trị vẫn được tính toán trong train_batch
    tb_logger.log_value('learnrate_pg', agent.optimizer.param_groups[0]['lr'], mini_step)            
    
    # total_cost không còn ý nghĩa, ta có thể log reward
    # avg_cost = (total_cost).mean().item()
    # tb_logger.log_value('train/avg_cost', avg_cost, mini_step)

    avg_reward_in_batch = torch.stack(reward, 0).mean().item()
    tb_logger.log_value('train/avg_reward', avg_reward_in_batch, mini_step)
    
    tb_logger.log_value('train/Target_Return', Reward.mean().item(), mini_step)
    tb_logger.log_value('train/ratios', ratios.mean().item(), mini_step)

    # initial_cost không còn ý nghĩa
    # tb_logger.log_value('train/init_cost', initial_cost.mean(), mini_step)

    grad_norms, grad_norms_clipped = grad_norms
    tb_logger.log_value('loss/actor_loss', reinforce_loss.item(), mini_step)
    tb_logger.log_value('loss/nll', -log_likelihood.mean().item(), mini_step)
    tb_logger.log_value('train/entropy', entropy.mean().item(), mini_step)
    # tb_logger.log_value('train/approx_kl_divergence', approx_kl_divergence.item(), mini_step) # Có thể gây lỗi nếu có inf
    tb_logger.log_histogram('train/bl_val',bl_val_detached.cpu(),mini_step)
    
    tb_logger.log_value('grad/actor', grad_norms[0].item(), mini_step)
    tb_logger.log_value('grad_clipped/actor', grad_norms_clipped[0].item(), mini_step)
    tb_logger.log_value('loss/critic_loss', baseline_loss.item(), mini_step)
            
    tb_logger.log_value('loss/total_loss', (reinforce_loss+baseline_loss).item(), mini_step)
    
    tb_logger.log_value('grad/critic', grad_norms[1].item(), mini_step)
    tb_logger.log_value('grad_clipped/critic', grad_norms_clipped[1].item(), mini_step)

# --- END OF FILE utils/logger.py ---