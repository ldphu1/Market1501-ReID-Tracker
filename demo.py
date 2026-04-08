import argparse
import cv2
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from ultralytics import YOLO
from model import *


def get_transform():
    return transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    #Trích xuất vector đặc trưng
def extract_embedding(model, img_array, transform, device):
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(img_rgb)

    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(img_tensor)
        emb = F.normalize(emb, p=2, dim=1)

    return emb.cpu()


def process_video(args, reid_model, detector, gallery_data, transform, device):
    gallery_embs = gallery_data["embs"]
    gallery_ids = gallery_data["ids"]

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Can't open video: {args.video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_path, fourcc, fps, (width, height))

    while True:
        flag, frame = cap.read()
        if not flag:
            break

        results = detector(frame, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        for box, cls in zip(boxes, classes):
            if int(cls) != 0:
                continue

            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            if x2 - x1 < 20 or y2 - y1 < 50:
                continue

            crop = frame[y1:y2, x1:x2]
            emb = extract_embedding(reid_model, crop, transform, device)

            sims = torch.matmul(emb, gallery_embs.T)

            best_idx = torch.argmax(sims, dim=1).item()
            score = sims[0, best_idx].item()
            pid = gallery_ids[best_idx]

            if score > args.threshold:
                label = f"ID: {pid} ({score:.2f})"
                color = (0, 255, 0)
            else:
                label = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        out.write(frame)

        cv2.imshow("Re-ID Tracking", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="data/video.avi",required=True, help="Path to the original video")
    parser.add_argument("--output_path", type=str, default="output_reid.mp4", help="Path to save the resulting video")
    parser.add_argument("--model_weights", type=str, default="weights/best_model.pth", help="Model weight file")
    parser.add_argument("--gallery_path", type=str, default="weights/gallery_market1501.pt", help="Path to gallery file")
    parser.add_argument("--threshold", type=float, default=0.6, help="threshold")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reid_model = resnet50(512)
    reid_model.load_state_dict(torch.load(args.model_weights, map_location=device))
    reid_model.to(device)
    reid_model.eval()

    gallery_data = torch.load(args.gallery_path)

    detector = YOLO("yolov8n.pt")

    transform = get_transform()

    process_video(args, reid_model, detector, gallery_data, transform, device)