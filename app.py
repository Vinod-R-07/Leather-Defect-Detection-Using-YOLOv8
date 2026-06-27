import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# ==================================
# PAGE SETTINGS
# ==================================
st.set_page_config(
    page_title="Leather Defect Detection",
    page_icon="🧵",
    layout="wide"
)

# ==================================
# LOAD MODEL
# ==================================
model = YOLO("runs/detect/train-3/weights/best.pt")

# ==================================
# TITLE
# ==================================
st.title("🧵 Leather Defect Detection Using YOLOv8")
st.markdown("Upload a leather image to automatically detect defects.")

# ==================================
# SIDEBAR
# ==================================
st.sidebar.header("Detection Settings")

display_confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.40,
    1.00,
    0.45,
    0.05
)

confidence = max(display_confidence - 0.4, 0.01)

# ==================================
# FILE UPLOAD
# ==================================
uploaded_file = st.file_uploader(
    "Upload Leather Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # ==================================
    # ORIGINAL IMAGE
    # ==================================
    image = Image.open(uploaded_file).convert("RGB")
    original_np = np.array(image)

    # ==================================
    # FIND LEATHER CONTOUR
    # ==================================
    gray = cv2.cvtColor(original_np, cv2.COLOR_RGB2GRAY)

    # Contrast Enhancement
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    _, thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    extracted_np = original_np.copy()

    ratio = 1.0

    if len(contours) > 0:

        largest_contour = max(
            contours,
            key=cv2.contourArea
        )

        largest_area = cv2.contourArea(
            largest_contour
        )

        image_area = (
            original_np.shape[0]
            * original_np.shape[1]
        )

        ratio = largest_area / image_area

        mask = np.zeros(
            gray.shape,
            dtype=np.uint8
        )

        cv2.drawContours(
            mask,
            [largest_contour],
            -1,
            255,
            thickness=cv2.FILLED
        )

    # ==================================
    # ADAPTIVE EXTRACTION + RESIZE
    # ==================================

    x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(
        largest_contour
    )

    coverage = (
        w_rect * h_rect
    ) / image_area

    st.write(
    "Coverage:",
    round(coverage, 3)
    )

    top = np.mean(mask[0, :] > 0)
    bottom = np.mean(mask[-1, :] > 0)
    left = np.mean(mask[:, 0] > 0)
    right = np.mean(mask[:, -1] > 0)

    full_leather = (
        coverage > 0.99
        or
        (
            top > 0.80
            and bottom > 0.80
            and left > 0.80
            and right > 0.80
        )
    )

    if full_leather:

        extracted_np = original_np.copy()
        resized_np = original_np.copy()

        extraction_status = "Skipped"
        resize_status = "Skipped"

    else:

        # Apply mask
        extracted_np = cv2.bitwise_and(
            original_np,
            original_np,
            mask=mask
        )

        # Get tight bounding box
        x, y, w, h = cv2.boundingRect(
            largest_contour
        )

        padding = 0

        x = max(0, x)
        y = max(0, y)

        w = min(
            original_np.shape[1] - x,
            w
        )

        h = min(
            original_np.shape[0] - y,
            h
        )

        # Crop only leather region
        cropped_leather = extracted_np[
            y:y+h,
            x:x+w
        ]

        # Resize cropped leather
        resized_np = cv2.resize(
            cropped_leather,
            (640, 640)
        )

        extraction_status = "Applied"
        resize_status = "Applied"

    # ==================================
    # DETECTION
    # ==================================
    with st.spinner("Detecting defects..."):

        results = model(
            resized_np,
            conf=confidence,
            imgsz=640
        )

        result = results[0]

        output = resized_np.copy()

        mask_resized = cv2.resize(
            mask,
            (640, 640)
        )

        dist_transform = cv2.distanceTransform(
            mask_resized,
            cv2.DIST_L2,
            5
        )

        BOUNDARY_DISTANCE = 20

        filtered_boxes = []

        for box in result.boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].cpu().numpy()
            )

            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )

            # Ignore boundary detections
            if (
                center_y < dist_transform.shape[0]
                and center_x < dist_transform.shape[1]
            ):

                if (
                    dist_transform[
                        center_y,
                        center_x
                    ] < BOUNDARY_DISTANCE
                ):
                    continue

            # Reduce box size
            shrink = 0.10

            width = x2 - x1
            height = y2 - y1

            x1 = int(
                x1 + width * shrink / 2
            )

            x2 = int(
                x2 - width * shrink / 2
            )

            y1 = int(
                y1 + height * shrink / 2
            )

            y2 = int(
                y2 - height * shrink / 2
            )

            filtered_boxes.append(box)

            conf_score = float(
                box.conf[0]
            )

            cls_id = int(
                box.cls[0]
            )

            class_name = model.names[
                cls_id
            ]

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                output,
                f"{class_name} {conf_score:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # ==================================
    # DISPLAY
    # ==================================
    st.subheader("Processing Pipeline")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### Original Image")
        st.image(
            original_np,
            use_container_width=True
        )
    with col2:

        st.markdown(
            f"### Leather Extraction + Contrast Enhancement"
        )

        st.image(
            extracted_np,
            use_container_width=True
        )

    with col3:

        st.markdown(
            f"### Resized Image"
        )

        st.image(
            resized_np,
            use_container_width=True
        )

    with col4:
        st.markdown(
            "### Detection Result"
        )

        st.image(
            output,
            use_container_width=True
        )

    st.success(
        "Detection completed successfully!"
    )

    # ==================================
    # DETECTED DEFECTS
    # ==================================
    st.subheader(
        "Detected Defects"
    )

    if len(filtered_boxes) == 0:

        st.warning(
            "No defects detected."
        )

    else:

        for box in filtered_boxes:

            cls_id = int(
                box.cls[0]
            )

            conf_score = float(
                box.conf[0]
            )

            class_name = model.names[
                cls_id
            ]

            st.write(
                f"**{class_name}** : {conf_score:.2f}"
            )