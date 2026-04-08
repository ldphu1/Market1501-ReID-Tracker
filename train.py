import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import sys
from model import *
from triplet_dataset import *
from eval_dataset import *

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform


def evaluate_rankk(model, query_loader, gallery_loader, device, k=5):
    model.eval()
    query_embs, query_ids = [], []
    gallery_embs, gallery_ids = [], []

    with torch.no_grad():
        for img, pid in tqdm(query_loader, desc="Extracting Query", leave=False, file=sys.stdout):
            emb = F.normalize(model(img.to(device)), p=2, dim=1)
            query_embs.append(emb.cpu())
            query_ids.extend(pid)

        for img, pid in tqdm(gallery_loader, desc="Extracting Gallery", leave=False, file=sys.stdout):
            emb = F.normalize(model(img.to(device)), p=2, dim=1)
            gallery_embs.append(emb.cpu())
            gallery_ids.extend(pid)

    query_embs = torch.cat(query_embs, dim=0)
    gallery_embs = torch.cat(gallery_embs, dim=0)

    sims = torch.matmul(query_embs, gallery_embs.T)
    topk_idx = torch.topk(sims, k=k, dim=1).indices

    correct = 0
    for i in range(len(query_ids)):
        if query_ids[i] in [gallery_ids[j] for j in topk_idx[i]]:
            correct += 1

    return correct / len(query_ids)

def train_epoch(model, dataloader, optimizer, criterion, device, epoch, writer, global_step):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", file=sys.stdout)
    for anchor, positive, negative in progress_bar:
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        # Forward
        emb_a = F.normalize(model(anchor), p=2, dim=1)
        emb_p = F.normalize(model(positive), p=2, dim=1)
        emb_n = F.normalize(model(negative), p=2, dim=1)

        loss = criterion(emb_a, emb_p, emb_n)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging to TensorBoard
        total_loss += loss.item()
        writer.add_scalar("Loss/train", loss.item(), global_step)
        global_step += 1

        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

    return total_loss / len(dataloader), global_step

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(args.save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Arguments: {args}")

    train_transform, val_transform = get_transforms()

    train_data = market1501(root=os.path.join(args.data_root, "bounding_box_train"), transform=train_transform)
    query_data = reiddts(root=os.path.join(args.data_root, "query"), transform=val_transform)
    gallery_data = reiddts(root=os.path.join(args.data_root, "bounding_box_test"), transform=val_transform)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              drop_last=True)
    query_loader = DataLoader(query_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    gallery_loader = DataLoader(gallery_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = resnet50_extractor(embedding_dim=512).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion = nn.TripletMarginLoss(margin=args.margin)

    global_step = 0
    best_rank1 = 0.0

    for epoch in range(1, args.epochs + 1):
        avg_loss, global_step = train_epoch(model, train_loader, optimizer, criterion, device, epoch, writer,
                                            global_step)
        print(f"Epoch [{epoch}/{args.epochs}] - Avg Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")

        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            print("Evaluating...")
            rank1 = evaluate_rankk(model, query_loader, gallery_loader, device, k=1)
            rank5 = evaluate_rankk(model, query_loader, gallery_loader, device, k=5)

            writer.add_scalar("Metric/Rank1", rank1, epoch)
            writer.add_scalar("Metric/Rank5", rank5, epoch)
            print(f"Rank-1: {rank1:.4f} | Rank-5: {rank5:.4f}")

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_rank1': best_rank1
            }
            torch.save(checkpoint, os.path.join(args.save_dir, f"model_epoch_{epoch}.pth"))

            if rank1 > best_rank1:
                best_rank1 = rank1
                torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
                print(f"Saved new best model with Rank-1: {best_rank1:.4f}")

        scheduler.step()

    torch.save(model.state_dict(), os.path.join(args.save_dir, "last_model.pth"))
    writer.close()
    print("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Person Re-ID Training Routine")

    parser.add_argument("--data_root", type=str, default="market1501",
                        help="Path to Market1501 dataset")
    parser.add_argument("--save_dir", type=str, default="weights", help="Directory to save models")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--margin", type=float, default=0.3, help="Margin for Triplet Loss")

    parser.add_argument("--step_size", type=int, default=10, help="Step size for LR Scheduler")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma for LR Scheduler")
    parser.add_argument("--eval_freq", type=int, default=5, help="Evaluate every N epochs")

    args = parser.parse_args()
    main(args)