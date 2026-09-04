# app.py — AI Structural Health Monitoring (SHM) Prototype
# This is the main Streamlit application file.
# Current version: UI, image upload, computer-vision preprocessing (grayscale,
# contrast enhancement, Gaussian blur, Canny edge detection), preliminary
# corrosion-candidate detection using HSV colour segmentation, experimental
# linear-anomaly screening, and a final preliminary inspection summary with a
# downloadable plain-text report.
# Defect classification (cracks, severity) is NOT yet implemented; corrosion
# candidates are rule-based screening results that require engineer verification.

import streamlit as st
import numpy as np
import cv2
from PIL import Image
from datetime import datetime  # used for the inspection timestamp in the summary and report

# ──────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Structural Health Monitoring",
    layout="wide",
)

# ──────────────────────────────────────────────
# Page title and introduction
# ──────────────────────────────────────────────
st.title(
    "Artificial Intelligence for Structural Health Monitoring (SHM) "
    "and Autonomous Robotic Inspection"
)

st.markdown(
    """
    This system supports visual inspection of critical civil and industrial infrastructure, including:

    - **Bridges** — decks, girders, piers, and abutments  
    - **Tunnels** — linings, joints, and portals  
    - **Pipelines** — external surfaces, welds, and supports  
    - **Industrial Structures** — tanks, frames, and pressure vessels  

    Upload an inspection image to get started.
    """
)

# ──────────────────────────────────────────────
# Sidebar — Inspection Settings
# ──────────────────────────────────────────────
st.sidebar.header("Inspection Settings")

structure_type = st.sidebar.selectbox(
    "Structure Type",
    options=["Pipeline", "Bridge", "Tunnel", "Industrial Structure"],
)

# Display the selected structure type for confirmation
st.sidebar.info(f"Selected structure type: **{structure_type}**")

# ──────────────────────────────────────────────
# Sidebar — Edge Detection Settings
# These sliders let the user tune the Canny edge detector thresholds.
# ──────────────────────────────────────────────
st.sidebar.header("Edge Detection Settings")

# Lower threshold: edges with gradient magnitude below this are discarded
lower_thresh = st.sidebar.slider(
    "Lower Canny Threshold",
    min_value=0,
    max_value=255,
    value=50,
)

# Upper threshold: edges with gradient magnitude above this are kept
upper_thresh = st.sidebar.slider(
    "Upper Canny Threshold",
    min_value=0,
    max_value=255,
    value=150,
)

# ──────────────────────────────────────────────
# Sidebar — Corrosion Detection Settings
# These sliders define the HSV colour window used to flag
# rust-coloured pixels as corrosion candidates.
# ──────────────────────────────────────────────
st.sidebar.header("Corrosion Detection Settings")

# Lower bound of the hue window (OpenCV hue range is 0–179)
min_hue = st.sidebar.slider(
    "Minimum Hue",
    min_value=0,
    max_value=179,
    value=0,
)

# Upper bound of the hue window (must exceed the minimum hue)
max_hue = st.sidebar.slider(
    "Maximum Hue",
    min_value=0,
    max_value=179,
    value=30,
)

# Saturation floor — greyish, desaturated pixels are unlikely to be rust
min_saturation = st.sidebar.slider(
    "Minimum Saturation",
    min_value=0,
    max_value=255,
    value=50,
)

# Brightness (value) floor — very dark pixels are unreliable
min_brightness = st.sidebar.slider(
    "Minimum Brightness",
    min_value=0,
    max_value=255,
    value=20,
)

# The hue window is invalid when the maximum is not greater than the
# minimum; warn immediately so the user can correct the sliders.
if max_hue <= min_hue:
    st.sidebar.warning(
        "Maximum hue must be greater than minimum hue for corrosion "
        "detection to run. Please adjust the hue sliders above."
    )

# ──────────────────────────────────────────────
# Sidebar — Crack Candidate Settings
# Rule-based screening for dark, elongated features
# that may be cracks. This is NOT trained AI.
# ──────────────────────────────────────────────
st.sidebar.header("Experimental Linear-Anomaly Settings")

# Smallest contour area (in pixels) accepted as a crack candidate.
# Contours smaller than this are treated as noise.
crack_min_area = st.sidebar.slider(
    "Minimum Contour Area (px)",
    min_value=5,
    max_value=500,
    value=15,
)

# Minimum elongation ratio (longer side / shorter side of the
# minimum-area bounding rectangle). A value of 1.0 means any shape
# is accepted; higher values restrict candidates to long, thin features.
crack_min_elongation = st.sidebar.slider(
    "Minimum Elongation Ratio",
    min_value=1.0,
    max_value=10.0,
    value=2.0,
    step=0.1,
)

# Largest acceptable contour area expressed as a percentage of the
# total image area. Very large regions are usually structural elements
# (e.g. joints, shadows) rather than cracks.
crack_max_area_pct = st.sidebar.slider(
    "Maximum Candidate Area (% of image)",
    min_value=0.1,
    max_value=10.0,
    value=3.0,
    step=0.1,
)

# ──────────────────────────────────────────────
# Image uploader
# ──────────────────────────────────────────────
st.header("Upload Inspection Image")

uploaded_file = st.file_uploader(
    "Choose an image file (JPG, JPEG, or PNG)",
    type=["jpg", "jpeg", "png"],
)

# ──────────────────────────────────────────────
# Display uploaded image or prompt user
# ──────────────────────────────────────────────
if uploaded_file is not None:
    # Open the image using Pillow and normalize it to RGB. Uploaded PNGs
    # may contain an alpha channel (RGBA); converting here guarantees a
    # 3-channel image for every downstream OpenCV operation.
    image = Image.open(uploaded_file).convert("RGB")

    # Show the original uploaded image with a heading
    st.subheader("Original Inspection Image")
    st.image(image, width="stretch")

    # Show basic file metadata
    st.write(
        f"**Filename:** {uploaded_file.name}  \n"
        f"**Dimensions:** {image.size[0]} × {image.size[1]} px"
    )

    # ──────────────────────────────────────────────
    # Analyze button — processing only runs when the user clicks this
    # ──────────────────────────────────────────────
    if st.button("Analyze Inspection Image", type="primary"):

        # ── Step 1: Validate Canny thresholds ──────────────────────────
        # The upper threshold must be greater than the lower threshold;
        # otherwise Canny will produce meaningless results.
        if upper_thresh <= lower_thresh:
            st.warning(
                "The upper Canny threshold must be greater than the lower "
                "threshold. Please adjust the sliders in the sidebar."
            )
        else:
            # ── Record the inspection date and time ──────────────────
            # datetime.now() captures the moment the analysis runs. The
            # Preliminary Inspection Summary and the downloadable report
            # both display this timestamp.
            inspection_datetime = datetime.now()

            # ── Step 2: Convert Pillow RGB image to a NumPy array ─────
            # Pillow gives us an RGB image; NumPy lets us work with pixel data.
            img_rgb = np.array(image)

            # Store original dimensions for the metrics table later
            orig_h, orig_w = img_rgb.shape[:2]

            # ── Step 3: Resize very large images ──────────────────────
            # If the image is wider than 1200 px, scale it down while
            # keeping the aspect ratio so processing stays fast.
            max_width = 1200
            if orig_w > max_width:
                scale = max_width / orig_w
                new_w = max_width
                new_h = int(orig_h * scale)
                img_rgb = cv2.resize(img_rgb, (new_w, new_h))

            # Record the dimensions we will actually process
            proc_h, proc_w = img_rgb.shape[:2]

            # ── Step 4: Convert RGB to BGR for OpenCV ─────────────────
            # OpenCV expects BGR channel order; Pillow provides RGB.
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            # ── Step 5: Convert to grayscale ──────────────────────────
            # Many edge-detection algorithms work on single-channel images.
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # ── Step 6: Enhance contrast with CLAHE ───────────────────
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            # improves local contrast without over-amplifying noise.
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_clahe = clahe.apply(img_gray)

            # ── Step 7: Apply Gaussian blur ───────────────────────────
            # A (5×5) Gaussian kernel smooths the image and reduces noise
            # before edge detection.
            img_blur = cv2.GaussianBlur(img_clahe, (5, 5), 0)

            # ── Step 8: Canny edge detection ──────────────────────────
            # Canny finds pixels where the intensity gradient is strong.
            # The two thresholds control hysteresis: weak edges connected
            # to strong edges are kept; isolated weak edges are discarded.
            edge_map = cv2.Canny(img_blur, lower_thresh, upper_thresh)

            # ── Step 9: Compute processing metrics ────────────────────
            # Count how many pixels were classified as edges.
            total_edge_pixels = int(np.count_nonzero(edge_map))
            total_pixels = proc_h * proc_w
            edge_percentage = (total_edge_pixels / total_pixels) * 100

            # ── Step 10: Display results in three columns ─────────────
            st.subheader("Processing Results")

            # Prepare the original (resized) image for display
            # by converting BGR back to RGB so colours look correct.
            display_original = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Original Image**")
                st.image(display_original, width="stretch")

            with col2:
                st.markdown("**Contrast-Enhanced Grayscale**")
                # CLAHE output is single-channel; Streamlit displays it as grayscale.
                st.image(img_clahe, width="stretch")

            with col3:
                st.markdown("**Edge Map**")
                # Canny output is single-channel (binary edge mask).
                st.image(edge_map, width="stretch")

                # Explanation under the edge map
                st.caption(
                    "The edge map highlights strong intensity changes. At this "
                    "stage, edges are possible defect candidates and must not be "
                    "interpreted as confirmed cracks."
                )

            # ── Step 11: Display processing metrics ───────────────────
            st.subheader("Processing Metrics")

            st.write(
                f"| Metric | Value |\n"
                f"|---|---|\n"
                f"| Original image width | {orig_w} px |\n"
                f"| Original image height | {orig_h} px |\n"
                f"| Processed image width | {proc_w} px |\n"
                f"| Processed image height | {proc_h} px |\n"
                f"| Total edge pixels | {total_edge_pixels:,} |\n"
                f"| Edge-pixel percentage | {edge_percentage:.2f}% |\n"
            )

            # ── Step 12: Corrosion-candidate detection ────────────────
            # Rule-based HSV colour segmentation flags rust-coloured
            # pixels as corrosion CANDIDATES. This is a screening aid,
            # not a defect classification.
            st.subheader("Corrosion Candidate Detection")

            # Initialize the corrosion result variables before the hue-window
            # check below. If the hue window is invalid, corrosion detection
            # is skipped and these placeholders let the final Preliminary
            # Inspection Summary run without errors.
            corrosion_metrics_available = False
            candidate_pixels = None
            candidate_area_pct = None
            accepted_region_count = None
            corrosion_extent = None

            if max_hue <= min_hue:
                # Invalid hue window — skip segmentation and explain why.
                st.warning(
                    "Maximum hue must be greater than minimum hue. "
                    "Please adjust the Corrosion Detection Settings in "
                    "the sidebar and analyze again."
                )
            else:
                # Confirm img_rgb is a three-channel RGB uint8 image.
                assert img_rgb.ndim == 3 and img_rgb.shape[2] == 3 and img_rgb.dtype == np.uint8, (
                    "img_rgb must be a 3-channel RGB uint8 image"
                )

                # ── HSV colour segmentation ─────────────────────────────
                # Convert the processed RGB image to the HSV colour space.
                # In OpenCV, hue spans 0–179; saturation and value span 0–255.
                hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

                # Keep pixels whose hue lies inside the selected window and
                # whose saturation and brightness exceed the thresholds.
                lower_bound = np.array(
                    [min_hue, min_saturation, min_brightness], dtype=np.uint8
                )
                upper_bound = np.array([max_hue, 255, 255], dtype=np.uint8)
                hsv_rust_mask = cv2.inRange(hsv, lower_bound, upper_bound)

                # ── RGB rust-dominance mask ──────────────────────────────
                # A pixel passes the RGB rust test only when red dominates
                # green, green dominates blue, and the red-blue gap is wide
                # enough — this rejects neutral greys and beiges that would
                # otherwise pass the HSV window alone.
                red = img_rgb[:, :, 0].astype(np.int16)
                green = img_rgb[:, :, 1].astype(np.int16)
                blue = img_rgb[:, :, 2].astype(np.int16)

                rgb_rust_mask = (
                    (red > green * 1.08)
                    & (green > blue * 1.05)
                    & ((red - blue) > 25)
                )

                # ── Combine masks with logical AND ──────────────────────
                # A pixel is a corrosion candidate only when it satisfies
                # BOTH the HSV rust-colour range AND the RGB red-brown
                # dominance rules.
                combined_mask_bool = (hsv_rust_mask > 0) & rgb_rust_mask
                combined_mask = (combined_mask_bool.astype(np.uint8)) * 255

                # Clean the combined mask with a 5×5 elliptical kernel:
                # opening removes small noise specks, closing fills small holes.
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask_opened = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
                mask_clean = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)

                # Find external contours of the cleaned mask.
                contours, _ = cv2.findContours(
                    mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                # Ignore contours smaller than 0.05% of the complete
                # image area — they are likely noise.
                min_region_area = 0.0005 * total_pixels
                accepted_contours = [
                    c for c in contours if cv2.contourArea(c) >= min_region_area
                ]

                # ── Step 13: Build the annotated image ────────────────
                # Blend a transparent orange overlay over the corrosion-
                # candidate pixels, then outline the accepted contours.
                # overlay and img_rgb are guaranteed to have the same height
                # and width, dtype uint8, and exactly three RGB channels,
                # because the uploaded image was normalized to RGB when
                # it was opened.
                orange_rgb = (255, 165, 0)
                overlay = img_rgb.copy()
                overlay[mask_clean > 0] = orange_rgb
                annotated = cv2.addWeighted(overlay, 0.4, img_rgb, 0.6, 0)
                cv2.drawContours(annotated, accepted_contours, -1, orange_rgb, 2)

                # ── Step 14: Display the mask and annotated image ──────
                col_corr1, col_corr2 = st.columns(2)

                with col_corr1:
                    st.markdown("**Corrosion Candidate Mask**")
                    st.image(mask_clean, width="stretch")
                    st.caption(
                        "White pixels satisfy both the HSV colour window "
                        "and RGB red-brown dominance after morphological cleanup."
                    )

                with col_corr2:
                    st.markdown("**Annotated Corrosion Candidates**")
                    st.image(annotated, width="stretch")
                    st.caption(
                        "Orange tint marks candidate pixels; outlines mark "
                        "accepted regions (at least 0.05% of the image area)."
                    )

                # ── Step 15: Corrosion metrics ─────────────────────────
                candidate_pixels = int(np.count_nonzero(mask_clean))
                candidate_area_pct = (candidate_pixels / total_pixels) * 100
                accepted_region_count = len(accepted_contours)

                # Classify the preliminary corrosion extent from the
                # candidate area percentage.
                if candidate_area_pct < 2.0:
                    corrosion_extent = "Minimal"
                elif candidate_area_pct < 10.0:
                    corrosion_extent = "Localized"
                elif candidate_area_pct < 25.0:
                    corrosion_extent = "Moderate"
                else:
                    corrosion_extent = "Extensive"

                # Corrosion results are complete, so the final Preliminary
                # Inspection Summary and the report may use them.
                corrosion_metrics_available = True

                st.subheader("Corrosion Detection Metrics")

                st.write(
                    f"| Metric | Value |\n"
                    f"|---|---|\n"
                    f"| Candidate pixel count | {candidate_pixels:,} |\n"
                    f"| Candidate area percentage | {candidate_area_pct:.2f}% |\n"
                    f"| Accepted region count | {accepted_region_count} |\n"
                )

                st.metric("Preliminary corrosion extent", corrosion_extent)
                st.caption(
                    "Extent bands based on candidate area: below 2% Minimal, "
                    "2% to below 10% Localized, 10% to below 25% Moderate, "
                    "25% or above Extensive."
                )

                # ── Dual-criteria explanation ─────────────────────────
                st.info(
                    "A pixel is flagged only when it satisfies both the "
                    "HSV rust-colour range and RGB red–brown dominance rules."
                )

                # ── Step 16: Limitations of colour segmentation ────────
                st.warning(
                    "**Colour segmentation limitation:** Corrosion candidates "
                    "come from rule-based HSV colour segmentation, which can "
                    "produce false positives due to lighting, paint, or soil. "
                    "Every flagged region requires verification by a qualified "
                    "engineer."
                )

            # ── Step 17: Crack-candidate detection ─────────────────
            # Rule-based screening that combines dark-feature morphology
            # with Canny edges to flag elongated dark features as crack
            # CANDIDATES. This is NOT trained AI and NOT confirmed crack
            # detection — it is a computer-vision screening aid.
            st.caption("*Research feature — not used for automated decisions*")
            st.subheader("Experimental Dark Linear-Anomaly Screening")

            # ── Morphological black-hat on contrast-enhanced grayscale ──
            # Black-hat highlights dark features (e.g. cracks, scratches)
            # that are surrounded by brighter pixels. The 21×21 elliptical
            # kernel gives dark linear features stronger local contrast.
            kernel_blackhat = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (21, 21)
            )
            blackhat = cv2.morphologyEx(
                img_clahe, cv2.MORPH_BLACKHAT, kernel_blackhat
            )

            # ── Normalize black-hat result to 0–255 ────────────────────
            # cv2.normalize stretches the pixel values so the darkest
            # feature maps to 0 and the brightest maps to 255.
            blackhat_norm = cv2.normalize(
                blackhat, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)

            # ── Otsu binary threshold ──────────────────────────────────
            # Otsu automatically picks the threshold that best separates
            # dark features from the background, producing a binary mask.
            _, dark_mask = cv2.threshold(
                blackhat_norm, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )

            # ── Dilate Canny edge map ─────────────────────────────────
            # A small 3×3 elliptical dilation thickens the edge map so
            # thin crack edges connect better with the dark-feature mask.
            small_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (3, 3)
            )
            dilated_edges = cv2.dilate(
                edge_map, small_kernel, iterations=1
            )

            # ── Combine dark-feature mask with dilated Canny edges ────
            # Keep only pixels that are BOTH dark (from black-hat) AND
            # on an edge (from dilated Canny). This suppresses dark
            # regions that have no edge structure (e.g. uniform shadows).
            dark_edge_mask = cv2.bitwise_and(dark_mask, dilated_edges)

            # ── Morphological closing ──────────────────────────────────
            # Closing with a 5×5 elliptical kernel for two iterations
            # bridges gaps between neighbouring crack-like pixels,
            # connecting broken pieces of the same crack.
            kernel_close = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (5, 5)
            )
            dark_edge_closed = cv2.morphologyEx(
                dark_edge_mask, cv2.MORPH_CLOSE, kernel_close,
                iterations=2,
            )

            # ── Dilate the crack-candidate mask ───────────────────────
            # A single dilation with the 3×3 kernel slightly expands the
            # connected mask so thin cracks remain visible for contour
            # detection.
            crack_mask = cv2.dilate(
                dark_edge_closed, small_kernel, iterations=1
            )

            # ── Find external contours ─────────────────────────────────
            # RETR_EXTERNAL retrieves only the outermost boundary of each
            # connected component — no nested holes.
            crack_contours, _ = cv2.findContours(
                crack_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            # ── Filter contours by area, elongation, and max size ──────
            # For each contour we check three rules:
            #   1. Area must be above the minimum (rejects noise).
            #   2. Elongation (long side / short side of the bounding
            #      rectangle) must meet the minimum (rejects round spots).
            #   3. Area must not exceed the maximum percentage of the
            #      whole image (rejects huge structural regions).
            crack_accepted = []
            max_crack_area = (crack_max_area_pct / 100.0) * total_pixels

            for cnt in crack_contours:
                # Rule 1: minimum area
                area = cv2.contourArea(cnt)
                if area < crack_min_area:
                    continue

                # Width and height from the minimum-area bounding rectangle
                _, (w_rect, h_rect), _ = cv2.minAreaRect(cnt)

                # Avoid division by zero if either dimension is zero
                if w_rect < 1e-6 or h_rect < 1e-6:
                    continue

                # Rule 2: elongation — longer side divided by shorter side
                longer = max(w_rect, h_rect)
                shorter = min(w_rect, h_rect)
                elongation = longer / shorter
                if elongation < crack_min_elongation:
                    continue

                # Rule 3: maximum area as percentage of total image
                if area > max_crack_area:
                    continue

                # All three rules passed — accept this crack candidate
                crack_accepted.append(cnt)

            # ── Draw accepted crack candidates in red ──────────────────
            # We draw on a copy of the original RGB image so corrosion
            # (orange) and crack (red) annotations stay separate.
            crack_annotated = img_rgb.copy()
            cv2.drawContours(
                crack_annotated, crack_accepted, -1, (255, 0, 0), 3
            )

            # ── Display dark-feature mask and annotated image ──────────
            col_crack1, col_crack2 = st.columns(2)

            with col_crack1:
                st.markdown("**Dark Linear Feature Mask**")
                st.image(crack_mask, width="stretch")
                st.caption(
                    "White pixels survived black-hat filtering, Otsu "
                    "thresholding, Canny-edge intersection, and "
                    "morphological closing."
                )

            with col_crack2:
                st.markdown("**Annotated Linear-Anomaly Candidates**")
                st.image(crack_annotated, width="stretch")
                st.caption(
                    "Red outlines mark dark, elongated features that "
                    "passed the rule-based screening criteria."
                )

            # ── Crack-candidate metrics ────────────────────────────────
            crack_count = len(crack_accepted)
            crack_pixel_area = int(
                sum(cv2.contourArea(c) for c in crack_accepted)
            )
            crack_area_pct = (crack_pixel_area / total_pixels) * 100

            st.subheader("Linear-Anomaly Candidate Metrics")

            st.write(
                f"| Metric | Value |\n"
                f"|---|---|\n"
                f"| Accepted linear-anomaly count | {crack_count} |\n"
                f"| Linear-anomaly pixel area | {crack_pixel_area:,} |\n"
                f"| Linear-anomaly area percentage | {crack_area_pct:.2f}% |\n"
            )

            # ── Rule-based screening warning ───────────────────────────
            st.warning(
                "Experimental output: This rule-based method identifies dark "
                "elongated visual features, but it cannot reliably distinguish "
                "cracks from joints, scratches, shadows, boundaries, or surface "
                "texture. These results are excluded from condition assessment "
                "and require engineer verification. A trained and validated "
                "crack-detection model is planned for the next development stage."
            )

            # ── Step 18: Preliminary Inspection Summary ─────────────
            # This final section gathers every key result from the analysis
            # into one overview, derives a review priority from the
            # preliminary corrosion extent alone, and prepares a plain-text
            # report the user can download. Nothing above this point is
            # changed by this section.
            st.subheader("Preliminary Inspection Summary")

            # Format the captured analysis time for display.
            inspection_datetime_str = inspection_datetime.strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            # Corrosion numbers exist only when the hue window was valid.
            # When detection was skipped, show a clear placeholder instead
            # of empty values.
            if corrosion_metrics_available:
                corrosion_area_summary = f"{candidate_area_pct:.2f}%"
                corrosion_count_summary = str(accepted_region_count)
                corrosion_extent_summary = corrosion_extent
            else:
                corrosion_area_summary = "Not available (invalid hue settings)"
                corrosion_count_summary = "Not available (invalid hue settings)"
                corrosion_extent_summary = "Not available (invalid hue settings)"

            # Overview table with the key facts of this inspection.
            st.write(
                f"| Item | Value |\n"
                f"|---|---|\n"
                f"| Inspection date and time | {inspection_datetime_str} |\n"
                f"| Uploaded filename | {uploaded_file.name} |\n"
                f"| Selected structure type | {structure_type} |\n"
                f"| Processed image dimensions | {proc_w} × {proc_h} px |\n"
                f"| Edge-pixel percentage | {edge_percentage:.2f}% |\n"
                f"| Corrosion candidate area percentage | {corrosion_area_summary} |\n"
                f"| Accepted corrosion-region count | {corrosion_count_summary} |\n"
                f"| Preliminary corrosion extent | {corrosion_extent_summary} |\n"
                f"| Experimental linear-anomaly count | {crack_count} |\n"
                f"| Experimental linear-anomaly area percentage | {crack_area_pct:.2f}% |\n"
            )

            # Linear-anomaly candidates are experimental and are deliberately
            # excluded from the review priority decided below.
            st.caption(
                "Linear-anomaly candidates are experimental screening "
                "results, are not confirmed cracks, and are not used to "
                "determine the review priority."
            )

            # ── Review priority ──────────────────────────────────────
            # The priority maps ONLY the preliminary corrosion extent to a
            # follow-up scheduling hint. It is not a structural severity
            # rating, remaining-life estimate, failure probability, or an
            # automated maintenance decision.
            priority_by_extent = {
                "Minimal": "Routine monitoring",
                "Localized": "Schedule engineer review",
                "Moderate": "Prioritized engineer and NDT review",
                "Extensive": "Prompt engineer and NDT assessment",
            }

            # Each priority band has one matching recommended action so the
            # on-screen message and the report always agree.
            action_by_extent = {
                "Minimal": (
                    "Record the result and continue planned monitoring."
                ),
                "Localized": (
                    "Document the candidate regions and schedule visual "
                    "verification."
                ),
                "Moderate": (
                    "Prioritize inspection by a qualified engineer and an "
                    "appropriate NDT method."
                ),
                "Extensive": (
                    "Arrange prompt qualified engineering assessment and "
                    "calibrated NDT."
                ),
            }

            if corrosion_metrics_available:
                review_priority = priority_by_extent[corrosion_extent]
                recommended_action = action_by_extent[corrosion_extent]
            else:
                # Without corrosion results no priority can be derived.
                review_priority = "Not determined (corrosion results unavailable)"
                recommended_action = (
                    "Correct the corrosion hue settings and analyze again "
                    "to obtain a review priority."
                )

            st.metric("Review priority", review_priority)

            # Required caveat, shown directly below the priority.
            st.caption(
                "Review priority is preliminary and is based only on visible "
                "corrosion-colour candidate area. It does not account for "
                "defect depth, wall-thickness loss, structural loading, or "
                "ultrasonic measurements."
            )

            # Clearly state what the review priority is NOT. It is only a
            # follow-up scheduling hint, never an engineering rating of
            # the structure, so this is spelled out on screen as well.
            st.caption(
                "The review priority is not a measure of structural severity, "
                "remaining service life, failure probability, or an automated "
                "engineering decision."
            )

            # Required limitation on what the review priority is based on.
            # The prototype has no depth, thickness, load, or ultrasonic
            # measurements, so it cannot account for any of them.
            st.caption(
                "The review priority is based only on visible corrosion-colour "
                "candidate area. The prototype does not measure corrosion "
                "depth, wall-thickness loss, structural load, or ultrasonic "
                "indications."
            )

            # Show the single recommended next action for this priority.
            st.markdown(
                f"**Recommended next action:** {recommended_action}"
            )

            # Required prototype statements. These two sentences must appear
            # word-for-word both here in the on-screen summary and inside the
            # downloaded plain-text report below.
            st.caption(
                "This prototype uses explainable rule-based computer vision. "
                "It is not a trained or validated defect-diagnosis model."
            )
            st.caption(
                "This report supports preliminary screening only and does not "
                "replace inspection by qualified engineers or certified NDT "
                "personnel."
            )

            # ── Step 19: Build the downloadable plain-text report ──
            # The report is one long text string assembled from the values
            # computed above. It records the settings, metrics, results, and
            # disclaimers — and deliberately contains NO image data.
            report_id = inspection_datetime.strftime("SHM-%Y%m%d-%H%M%S")

            # Corrosion block for the report: real numbers when available,
            # an explanation when detection was skipped.
            if corrosion_metrics_available:
                corrosion_report_block = (
                    f"Candidate pixel count: {candidate_pixels:,}\n"
                    f"Candidate area percentage: {candidate_area_pct:.2f}%\n"
                    f"Accepted region count: {accepted_region_count}\n"
                    f"Preliminary corrosion extent: {corrosion_extent}\n"
                    "Extent bands: below 2% Minimal, 2% to below 10% Localized,\n"
                    "10% to below 25% Moderate, 25% or above Extensive."
                )
            else:
                corrosion_report_block = (
                    "Corrosion detection was not performed because the hue\n"
                    "window was invalid (maximum hue must be greater than\n"
                    "minimum hue). Adjust the corrosion settings and analyze\n"
                    "again to obtain corrosion results."
                )

            # Horizontal rules keep the plain-text report readable in any
            # text editor.
            thick_rule = "=" * 72
            thin_rule = "-" * 72

            report_text = (
                thick_rule + "\n"
                + "Artificial Intelligence for Structural Health Monitoring (SHM)\n"
                + "and Autonomous Robotic Inspection\n"
                + "Preliminary Inspection Report\n"
                + thick_rule + "\n\n"
                + f"Report ID: {report_id}\n"
                + f"Inspection date and time: {inspection_datetime_str}\n"
                + f"Uploaded filename: {uploaded_file.name}\n"
                + f"Structure type: {structure_type}\n\n"
                + thin_rule + "\n"
                + "PROCESSING SETTINGS\n"
                + thin_rule + "\n"
                + f"Lower Canny threshold: {lower_thresh}\n"
                + f"Upper Canny threshold: {upper_thresh}\n"
                + f"Minimum hue: {min_hue}\n"
                + f"Maximum hue: {max_hue}\n"
                + f"Minimum saturation: {min_saturation}\n"
                + f"Minimum brightness: {min_brightness}\n"
                + f"Minimum linear-anomaly contour area: {crack_min_area} px\n"
                + f"Minimum linear-anomaly elongation ratio: {crack_min_elongation}\n"
                + f"Maximum linear-anomaly candidate area: {crack_max_area_pct}% of image\n"
                + "Maximum processing width (resize cap): 1200 px\n"
                + "CLAHE clip limit / tile grid: 2.0 / 8x8\n"
                + "Gaussian blur kernel: 5x5\n"
                + "Corrosion mask cleanup: 5x5 elliptical opening then closing\n"
                + "Minimum accepted corrosion region: 0.05% of image area\n"
                + "Linear-anomaly black-hat kernel: 21x21 elliptical\n"
                + "Linear-anomaly fusion: dark mask AND 3x3-dilated Canny edges,\n"
                + "then 5x5 elliptical closing (2 iterations) and 3x3 dilation\n\n"
                + thin_rule + "\n"
                + "PROCESSING METRICS\n"
                + thin_rule + "\n"
                + f"Original image width: {orig_w} px\n"
                + f"Original image height: {orig_h} px\n"
                + f"Processed image width: {proc_w} px\n"
                + f"Processed image height: {proc_h} px\n"
                + f"Total edge pixels: {total_edge_pixels:,}\n"
                + f"Edge-pixel percentage: {edge_percentage:.2f}%\n\n"
                + thin_rule + "\n"
                + "CORROSION RESULTS\n"
                + thin_rule + "\n"
                + corrosion_report_block + "\n\n"
                + thin_rule + "\n"
                + "EXPERIMENTAL LINEAR-ANOMALY RESULTS\n"
                + thin_rule + "\n"
                + f"Accepted linear-anomaly count: {crack_count}\n"
                + f"Linear-anomaly pixel area: {crack_pixel_area:,}\n"
                + f"Linear-anomaly area percentage: {crack_area_pct:.2f}%\n"
                + "Note: Linear-anomaly candidates are experimental screening\n"
                + "results. They are not confirmed cracks and are not used to\n"
                + "determine the review priority.\n\n"
                + thin_rule + "\n"
                + "PRELIMINARY ASSESSMENT\n"
                + thin_rule + "\n"
                + f"Review priority: {review_priority}\n"
                + f"Recommended next action: {recommended_action}\n\n"
                + "The review priority is not a measure of structural severity,\n"
                + "remaining service life, failure probability, or an automated\n"
                + "engineering decision.\n\n"
                + "The review priority is based only on visible corrosion-colour candidate area. The prototype does not measure corrosion depth, wall-thickness loss, structural load, or ultrasonic indications.\n\n"
                + "Review priority is preliminary and is based only on visible\n"
                + "corrosion-colour candidate area. It does not account for defect\n"
                + "depth, wall-thickness loss, structural loading, or ultrasonic\n"
                + "measurements.\n\n"
                + thin_rule + "\n"
                + "METHOD LIMITATIONS\n"
                + thin_rule + "\n"
                + "This prototype uses explainable rule-based computer vision. It is not a trained or validated defect-diagnosis model.\n\n"
                + "- Corrosion candidates come from rule-based HSV colour\n"
                + "  segmentation fused with RGB red-brown dominance rules, which\n"
                + "  can produce false positives due to lighting, paint, or soil,\n"
                + "  and can miss corrosion that is not rust-coloured.\n"
                + "- The experimental linear-anomaly screening cannot reliably\n"
                + "  distinguish cracks from joints, scratches, shadows,\n"
                + "  boundaries, or surface texture.\n"
                + "- Edge, corrosion, and linear-anomaly outputs are candidate\n"
                + "  indications only; every flagged region requires verification\n"
                + "  by a qualified engineer.\n\n"
                + thin_rule + "\n"
                + "ENGINEER-VERIFICATION DISCLAIMER\n"
                + thin_rule + "\n"
                + "This report supports preliminary screening only and does not replace inspection by qualified engineers or certified NDT personnel.\n"
                + "All candidate findings must be verified on site by a qualified engineer using appropriate inspection methods.\n\n"
                + thin_rule + "\n"
                + "REPORT NOTES\n"
                + thin_rule + "\n"
                + "This report contains no image data; only the filename,\n"
                + "settings, metrics, and screening results are recorded.\n\n"
                + thin_rule + "\n"
                + "Prepared by: Unzila Zeb, Momina Rizwan, Naba Raheel\n"
                + "Institution: NUST CEME\n"
                + thick_rule + "\n"
            )

            # ── Step 20: Report download button ─────────────────────
            # st.download_button turns the report string into a file the
            # browser saves locally. The filename embeds the analysis
            # timestamp (YYYYMMDD_HHMMSS) so every report is uniquely named,
            # and mime="text/plain" tells the browser it is a plain-text
            # file that opens in any text editor.
            report_filename = (
                "inspection_report_"
                + inspection_datetime.strftime("%Y%m%d_%H%M%S")
                + ".txt"
            )

            st.download_button(
                label="Download Preliminary Inspection Report",
                data=report_text,
                file_name=report_filename,
                mime="text/plain",
            )

            st.caption(
                "The report is plain text, contains no image data, and is "
                "intended for preliminary screening records only."
            )

else:
    # No image uploaded yet — guide the user
    st.info("Please upload an inspection image to begin analysis.")

# ──────────────────────────────────────────────
# Disclaimer
# ──────────────────────────────────────────────
st.markdown("---")
st.warning(
    "**Disclaimer:** This prototype is designed to support engineers in preliminary "
    "assessment and does **not** replace certified Non-Destructive Testing (NDT) "
    "inspection by qualified professionals."
)
