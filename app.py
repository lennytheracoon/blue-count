import cv2
import streamlit as st
import numpy as np
import supervision as sv
from inference import get_model

st.set_page_config(
    page_title="Blueberry Counter",
    page_icon="🫐",
    layout="wide"
)

# Function to count all detections from inference results
def count_detections(results):
    total_count = 0
    for result in results:
        total_count += len(result.predictions)
    return total_count

def display_humor(detection_count):
    if detection_count == 0:
        st.error("No blueberries found. Looks like the raccoons beat you to them!")
    elif detection_count == 1:
        st.success("Only one blueberry found. That's a raccoon-sized snack!")
    else:
        st.info(f"{detection_count} blueberries found. Looks like you hit the jackpot!")

# Load model
model = get_model(model_id="blueberry-3ewrk/1")

# Main function to run the Streamlit app
def main():
    st.title("🫐Blueberry Counter")

    # Upload image
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        results = model.infer(image_rgb)

        # Count all detections
        detection_count = count_detections(results)

        # Load the results into the supervision Detections API
        detections = sv.Detections.from_roboflow(results[0].dict(by_alias=True, exclude_none=True))

        # Create supervision annotators
        bounding_box_annotator = sv.BoundingBoxAnnotator()

        # Annotate the image with inference results
        annotated_image = bounding_box_annotator.annotate(scene=image_rgb, detections=detections)

        # Display the image and detection count side by side
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(annotated_image,  use_column_width=True)
        with col2:
            display_humor(detection_count)

if __name__ == "__main__":
    main()
