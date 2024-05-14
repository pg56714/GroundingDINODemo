import cv2
import gradio as gr
import numpy as np
from PIL import Image
import warnings

warnings.filterwarnings("ignore")
from groundingdino.util.inference import annotate, predict, load_model
import groundingdino.datasets.transforms as T

MARKDOWN = """
# GroundingDINODemo

Powered by [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO).
"""

config_file = "./groundingdino/config/GroundingDINO_SwinT_OGC.py"
ckpt_filenmae = "./weights/groundingdino_swint_ogc.pth"


def image_transform_grounding(init_image):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(init_image, None)
    return init_image, image


def image_transform_grounding_for_vis(init_image):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
        ]
    )
    image, _ = transform(init_image, None)
    return image


model = load_model(config_file, ckpt_filenmae)


def run_grounding(input_image, grounding_caption, box_threshold, text_threshold):
    init_image = Image.fromarray(input_image.astype("uint8"), "RGB")

    _, image_tensor = image_transform_grounding(init_image)
    image_pil: Image = image_transform_grounding_for_vis(init_image)

    boxes, logits, phrases = predict(
        model,
        image_tensor,
        grounding_caption,
        box_threshold,
        text_threshold,
        device="cpu",
    )
    annotated_frame = annotate(
        image_source=np.asarray(image_pil), boxes=boxes, logits=logits, phrases=phrases
    )
    image_with_box = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

    return image_with_box


# View
with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    with gr.Row():
        with gr.Column():
            input_image_component = gr.Image(type="numpy", label="Input Image")

            with gr.Accordion("GroundingDINO", open=False):
                box_threshold = gr.Slider(
                    label="Box Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.35,
                    step=0.001,
                )
                text_threshold = gr.Slider(
                    label="Text Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.25,
                    step=0.001,
                )

            image_categories_text_component = gr.Textbox(
                label="Categories",
                placeholder="comma separated list of categories",
                scale=7,
            )

        grounding_dion_output_image_component = gr.Image(
            type="pil", label="GroundingDINO Output"
        )
    submit_button_component = gr.Button(value="Submit", scale=1, variant="primary")

    submit_button_component.click(
        fn=run_grounding,
        inputs=[
            input_image_component,
            image_categories_text_component,
            box_threshold,
            text_threshold,
        ],
        outputs=[grounding_dion_output_image_component],
    )

demo.launch(debug=False, show_error=True, max_threads=1)
