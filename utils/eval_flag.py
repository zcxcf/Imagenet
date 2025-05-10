import torch
from tqdm import tqdm
import wandb
import torch.distributed as dist

def eval_mat_combined_dis(model, valDataLoader, criterion, epoch, optimizer, args, flag, device, local_rank):

    with tqdm(total=len(valDataLoader), postfix=dict, mininterval=0.3) as pbar:
        pbar.set_description(f'eval Epoch {epoch + 1}/{args.epochs}')

        model.eval()

        with torch.no_grad():
            total_loss = 0.0
            correct = 0
            total = 0
            for batch_idx, (img, label) in enumerate(valDataLoader):
                img = img.to(device)
                label = label.to(device)

                preds = model(img)

                loss = criterion(preds, label)
                total_loss += loss.item()

                _, predicted = torch.max(preds, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                pbar.set_postfix(**{"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
                pbar.update(1)

            correct = torch.tensor(correct, dtype=torch.float32, device='cuda')
            total = torch.tensor(total, dtype=torch.float32, device='cuda')

            dist.all_reduce(correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(total, op=dist.ReduceOp.SUM)

            if local_rank==0:
                accuracy = correct / total
                print(f"Global Accuracy: {accuracy.item()}")

                wandb.log({"Epoch": epoch + 1, "Val Acc_" + str(local_rank) + flag: accuracy.item()})


def eval_router_dis(model, valDataLoader, criterion, epoch, optimizer, args, flag, device, local_rank):
    with tqdm(total=len(valDataLoader), postfix=dict, mininterval=0.3) as pbar:
        pbar.set_description(f'eval Epoch {epoch + 1}/{args.epochs}')

        model.eval()

        with torch.no_grad():
            total_loss = 0.0
            correct = 0
            total = 0
            total_macs_sum = 0

            for batch_idx, (img, label) in enumerate(valDataLoader):
                img = img.to(device)
                label = label.to(device)

                preds, attn_mask, mlp_mask, embed_mask, depth_attn_mask, depth_mlp_mask, total_macs = model(img)

                total_macs_sum += total_macs.item()

                loss = criterion(preds, label)
                total_loss += loss.item()

                _, predicted = torch.max(preds, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                pbar.set_postfix(**{"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
                pbar.update(1)

            correct = torch.tensor(correct, dtype=torch.float32, device='cuda')
            total = torch.tensor(total, dtype=torch.float32, device='cuda')

            dist.all_reduce(correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(total, op=dist.ReduceOp.SUM)

            if local_rank == 0:
                val_macs = total_macs_sum / len(valDataLoader)

                accuracy = correct / total
                print(f"Global Accuracy: {accuracy.item()}")

                wandb.log({"Epoch": epoch + 1, "Val Acc_" + str(local_rank) + flag: accuracy.item()})
                wandb.log({"Epoch": epoch + 1, "val_mac/Val macs_" + str(local_rank) + flag: val_macs})





