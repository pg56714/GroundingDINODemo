from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

CONFIG_PATH = "./groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "./weights/groundingdino_swint_ogc.pth"
DEVICE = "cuda"
IMAGE_PATH = "./test.jpg"
TEXT_PROMPT = "person"
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25
image_source, image = load_image(IMAGE_PATH)
model = load_model(CONFIG_PATH, CHECKPOINT_PATH)
boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD,
    device=DEVICE,
)
annotated_frame = annotate(
    image_source=image_source, boxes=boxes, logits=logits, phrases=phrases
)
cv2.imwrite(
    "./annotated_image.jpg",
    annotated_frame,
)
