from transformers import AutoFeatureExtractor, AutoModelForObjectDetection
from PIL import Image
import torch
import matplotlib.pyplot as plt
# import tensorflow as tf
import io
COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933]
]


extractor = AutoFeatureExtractor.from_pretrained("facebook/detr-resnet-50")

model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")


def fig2img(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def visualize_prediction(pil_img, output_dict, threshold=0.7, id2label=None):
    keep = output_dict["scores"] > threshold
    boxes = output_dict["boxes"][keep].tolist()
    scores = output_dict["scores"][keep].tolist()
    labels = output_dict["labels"][keep].tolist()
    if id2label is not None:
        labels = [id2label[x] for x in labels]

    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, (xmin, ymin, xmax, ymax), label, color in zip(scores, boxes, labels, colors):
        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin), (xmax - xmin), (ymax - ymin), fill=False, color=color, linewidth=3
            )
        )
        ax.text(
            xmin, ymin, f"{label}: {score:0.2f}", fontsize=12, bbox=dict(
                facecolor="yellow", alpha=0.5
            )
        )
    plt.axis("off")
    return fig2img(plt.gcf()), labels


def main(image, threshold):
    inputs = extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    img_size = torch.tensor([tuple(reversed(image.size))])
    processed_outputs = extractor.post_process(outputs, img_size)[0]
    imageEnd, Labels = visualize_prediction(
        image,
        processed_outputs,
        threshold,
        model.config.id2label
    )
    print(Labels)
    return imageEnd, Labels


if __name__ == "__main__":
    img, labels = main(Image.open("./images/4.jpeg"))
    print(img.show())
