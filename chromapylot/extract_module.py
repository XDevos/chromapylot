from skimage import measure


def extract_properties(tiff_image, mask):
    # Measure properties of labeled image regions
    properties = measure.regionprops_table(
        mask,
        intensity_image=tiff_image,
        properties=(
            "label",
            "max_intensity",
        ),
    )
    return properties
