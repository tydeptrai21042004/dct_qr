from __future__ import annotations

from .types import AttackConfig


def _a(attack_id: str, group: str, params: dict, category: str, severity: str) -> AttackConfig:
    return AttackConfig(attack_id, group, params, category, severity)


SANITY = (
    _a("none__clean", "none", {}, "clean", "none"),
    _a("jpeg__q90", "jpeg", {"quality": 90}, "compression", "mild"),
    _a("noise_gaussian__sigma1", "gaussian_noise", {"sigma": 1.0, "seed": 123}, "noise", "mild"),
    _a("filter_gaussian__radius0p5", "gaussian_blur", {"radius": 0.5}, "filtering", "mild"),
    _a("photometric_brightness__factor0p95", "brightness", {"factor": 0.95}, "photometric", "mild"),
    _a("geometric_resize__factor0p9", "resize", {"factor": 0.9}, "geometric", "mild"),
)

COMPRESSION = (
    _a("jpeg__q95", "jpeg", {"quality": 95}, "compression", "mild"),
    _a("jpeg__q90", "jpeg", {"quality": 90}, "compression", "mild"),
    _a("jpeg__q80", "jpeg", {"quality": 80}, "compression", "moderate"),
    _a("jpeg__q70", "jpeg", {"quality": 70}, "compression", "moderate"),
    _a("jpeg__q50", "jpeg", {"quality": 50}, "compression", "strong"),
    _a("jpeg__q30", "jpeg", {"quality": 30}, "compression", "strong"),
    _a("webp__q90", "webp", {"quality": 90}, "compression", "mild"),
    _a("webp__q70", "webp", {"quality": 70}, "compression", "moderate"),
    _a("webp__q50", "webp", {"quality": 50}, "compression", "strong"),
    _a("jpeg2000__layer7", "jpeg2000", {"quality_layer": 7.0}, "compression", "moderate"),
    _a("chroma_subsampling__factor2", "chroma_subsampling", {"factor": 2}, "compression", "moderate"),
    _a("chroma_subsampling__factor4", "chroma_subsampling", {"factor": 4}, "compression", "strong"),
    _a("color_quantization__colors64", "color_quantization", {"colors": 64}, "compression", "moderate"),
    _a("bit_depth__bits5", "bit_depth", {"bits": 5}, "compression", "moderate"),
    _a("bit_depth__bits3", "bit_depth", {"bits": 3}, "compression", "strong"),
)

NOISE = (
    _a("noise_gaussian__sigma1", "gaussian_noise", {"sigma": 1.0, "seed": 123}, "noise", "mild"),
    _a("noise_gaussian__sigma3", "gaussian_noise", {"sigma": 3.0, "seed": 123}, "noise", "moderate"),
    _a("noise_gaussian__sigma5", "gaussian_noise", {"sigma": 5.0, "seed": 123}, "noise", "strong"),
    _a("noise_gaussian_var__0p001", "gaussian_noise_var", {"variance": 0.001, "seed": 123}, "noise", "mild"),
    _a("noise_gaussian_var__0p003", "gaussian_noise_var", {"variance": 0.003, "seed": 123}, "noise", "strong"),
    _a("noise_salt_pepper__0p001", "salt_pepper", {"amount": 0.001, "seed": 123}, "noise", "mild"),
    _a("noise_salt_pepper__0p005", "salt_pepper", {"amount": 0.005, "seed": 123}, "noise", "moderate"),
    _a("noise_salt_pepper__0p01", "salt_pepper", {"amount": 0.01, "seed": 123}, "noise", "strong"),
    _a("noise_speckle__var0p005", "speckle_noise", {"variance": 0.005, "seed": 123}, "noise", "moderate"),
    _a("noise_poisson__peak255", "poisson_noise", {"peak": 255.0, "seed": 123}, "noise", "mild"),
    _a("noise_poisson__peak64", "poisson_noise", {"peak": 64.0, "seed": 123}, "noise", "strong"),
    _a("pixel_dropout__amount0p01", "pixel_dropout", {"amount": 0.01, "seed": 123}, "noise", "moderate"),
)

FILTERING = (
    _a("filter_median__size3", "median_filter", {"size": 3}, "filtering", "moderate"),
    _a("filter_median__size5", "median_filter", {"size": 5}, "filtering", "strong"),
    _a("filter_average__size3", "average_filter", {"size": 3}, "filtering", "moderate"),
    _a("filter_average__size5", "average_filter", {"size": 5}, "filtering", "strong"),
    _a("filter_gaussian__radius0p5", "gaussian_blur", {"radius": 0.5}, "filtering", "mild"),
    _a("filter_gaussian__radius1", "gaussian_blur", {"radius": 1.0}, "filtering", "moderate"),
    _a("filter_motion__size7_horizontal", "motion_blur", {"size": 7, "angle": "horizontal"}, "filtering", "moderate"),
    _a("filter_motion__size9_vertical", "motion_blur", {"size": 9, "angle": "vertical"}, "filtering", "strong"),
    _a("filter_bilateral__d7", "bilateral_filter", {"diameter": 7, "sigma_color": 50, "sigma_space": 50}, "filtering", "moderate"),
    _a("filter_unsharp__r2_p150", "unsharp_mask", {"radius": 2.0, "percent": 150, "threshold": 3}, "filtering", "moderate"),
    _a("filter_sharpen__factor2", "sharpen", {"factor": 2.0}, "filtering", "moderate"),
)

GEOMETRIC = (
    _a("geometric_rotation__deg1", "rotation", {"degrees": 1.0}, "geometric", "mild"),
    _a("geometric_rotation__deg2", "rotation", {"degrees": 2.0}, "geometric", "moderate"),
    _a("geometric_rotation__deg5", "rotation", {"degrees": 5.0}, "geometric", "strong"),
    _a("geometric_rotation_back__deg1", "rotation_back", {"degrees": 1.0}, "geometric", "mild"),
    _a("geometric_translation__x4_y4", "translation", {"shift_x": 4, "shift_y": 4}, "geometric", "mild"),
    _a("geometric_translation__x8_y8", "translation", {"shift_x": 8, "shift_y": 8}, "geometric", "moderate"),
    _a("geometric_resize__factor0p75", "resize", {"factor": 0.75}, "geometric", "moderate"),
    _a("geometric_resize__factor1p5", "resize", {"factor": 1.5}, "geometric", "moderate"),
    _a("geometric_resample_cycle__0p67_x2", "resample_cycle", {"down_factor": 0.67, "cycles": 2}, "geometric", "strong"),
    _a("geometric_crop_resize__keep0p9", "crop_resize", {"keep": 0.9}, "geometric", "moderate"),
    _a("geometric_crop_resize__keep0p75", "crop_resize", {"keep": 0.75}, "geometric", "strong"),
    _a("geometric_shear__x0p05", "shear", {"shear_x": 0.05}, "geometric", "moderate"),
    _a("geometric_shear__x0p08", "shear", {"shear_x": 0.08}, "geometric", "strong"),
    _a("geometric_affine__r2_s0p98_sh0p02", "affine", {"rotation": 2.0, "scale": 0.98, "shear_x": 0.02}, "geometric", "strong"),
    _a("geometric_perspective__strength0p02", "perspective", {"strength": 0.02, "seed": 123}, "geometric", "moderate"),
    _a("geometric_perspective__strength0p04", "perspective", {"strength": 0.04, "seed": 123}, "geometric", "strong"),
    _a("geometric_row_col_delete__4_4", "row_col_delete", {"rows": 4, "cols": 4, "seed": 123}, "geometric", "moderate"),
    _a("geometric_border_crop_pad__px16", "border_crop_pad", {"pixels": 16}, "geometric", "moderate"),
)

PHOTOMETRIC = (
    _a("photometric_gamma__0p8", "gamma", {"gamma": 0.8}, "photometric", "moderate"),
    _a("photometric_gamma__1p2", "gamma", {"gamma": 1.2}, "photometric", "moderate"),
    _a("photometric_brightness__factor0p8", "brightness", {"factor": 0.8}, "photometric", "strong"),
    _a("photometric_brightness__factor1p2", "brightness", {"factor": 1.2}, "photometric", "strong"),
    _a("photometric_contrast__factor0p8", "contrast", {"factor": 0.8}, "photometric", "strong"),
    _a("photometric_contrast__factor1p2", "contrast", {"factor": 1.2}, "photometric", "strong"),
    _a("photometric_saturation__factor0p5", "saturation", {"factor": 0.5}, "photometric", "moderate"),
    _a("photometric_hue__deg15", "hue_shift", {"degrees": 15.0}, "photometric", "moderate"),
    _a("photometric_white_balance__warm", "white_balance", {"red": 1.08, "green": 1.0, "blue": 0.92}, "photometric", "moderate"),
    _a("photometric_hist_equalization", "hist_equalization", {}, "photometric", "strong"),
    _a("photometric_clahe_like", "clahe_like", {}, "photometric", "strong"),
    _a("photometric_channel_dropout__blue", "channel_dropout", {"channel": 2, "value": 0}, "photometric", "strong"),
    _a("photometric_adaptive_threshold", "adaptive_threshold", {"block_size": 31, "c": 3.0}, "photometric", "extreme"),
)

OCCLUSION = (
    _a("occlusion_random__block64", "occlusion", {"block": 64, "seed": 123}, "occlusion", "mild"),
    _a("occlusion_center__fraction0p1", "occlusion_fraction", {"fraction": 0.10}, "occlusion", "mild"),
    _a("occlusion_center__fraction0p25", "occlusion_fraction", {"fraction": 0.25}, "occlusion", "strong"),
    _a("occlusion_center__fraction0p5", "occlusion_fraction", {"fraction": 0.50}, "occlusion", "extreme"),
    _a("occlusion_checkerboard__tile32", "checkerboard_cutout", {"tile": 32}, "occlusion", "extreme"),
    _a("occlusion_mosaic__factor8", "mosaic", {"factor": 8}, "occlusion", "strong"),
)

COMBINED = (
    _a("combined__jpeg70_gaussian1", "combined", {"steps": [
        {"group": "jpeg", "params": {"quality": 70}},
        {"group": "gaussian_noise", "params": {"sigma": 1.0, "seed": 123}},
    ]}, "combined", "strong"),
    _a("combined__jpeg70_blur1", "combined", {"steps": [
        {"group": "jpeg", "params": {"quality": 70}},
        {"group": "gaussian_blur", "params": {"radius": 1.0}},
    ]}, "combined", "strong"),
    _a("combined__rotate2_jpeg70", "combined", {"steps": [
        {"group": "rotation", "params": {"degrees": 2.0}},
        {"group": "jpeg", "params": {"quality": 70}},
    ]}, "combined", "strong"),
    _a("combined__resize0p75_jpeg70", "combined", {"steps": [
        {"group": "resize", "params": {"factor": 0.75}},
        {"group": "jpeg", "params": {"quality": 70}},
    ]}, "combined", "strong"),
    _a("combined__crop0p9_jpeg70", "combined", {"steps": [
        {"group": "crop_resize", "params": {"keep": 0.9}},
        {"group": "jpeg", "params": {"quality": 70}},
    ]}, "combined", "strong"),
    _a("combined__perspective0p02_jpeg80", "combined", {"steps": [
        {"group": "perspective", "params": {"strength": 0.02, "seed": 123}},
        {"group": "jpeg", "params": {"quality": 80}},
    ]}, "combined", "strong"),
)

def _deduplicate(configs: tuple[AttackConfig, ...]) -> tuple[AttackConfig, ...]:
    seen: set[str] = set()
    output: list[AttackConfig] = []
    for cfg in configs:
        if cfg.attack_id not in seen:
            seen.add(cfg.attack_id)
            output.append(cfg)
    return tuple(output)


# Curated paper-facing suite: exactly 20 common, well-known image-processing
# attacks, grouped into five standard robustness categories (4 attacks each).
# The suite is intentionally moderate: it is meant for the main comparison
# table, while severe/compound/real-world attacks remain in stress/publication.
COMMON_20 = (
    # Compression / quantization
    _a("common20_jpeg__q70", "jpeg", {"quality": 70}, "compression", "moderate"),
    _a("common20_jpeg2000__layer7", "jpeg2000", {"quality_layer": 7.0}, "compression", "moderate"),
    _a("common20_bit_depth__bits6", "bit_depth", {"bits": 6}, "compression", "moderate"),
    _a("common20_color_quantization__colors64", "color_quantization", {"colors": 64}, "compression", "moderate"),

    # Noise
    _a("common20_gaussian_noise__sigma1", "gaussian_noise", {"sigma": 1.0, "seed": 123}, "noise", "mild"),
    _a("common20_salt_pepper__0p005", "salt_pepper", {"amount": 0.005, "seed": 123}, "noise", "moderate"),
    _a("common20_speckle__var0p005", "speckle_noise", {"variance": 0.005, "seed": 123}, "noise", "moderate"),
    _a("common20_poisson__peak255", "poisson_noise", {"peak": 255.0, "seed": 123}, "noise", "mild"),

    # Filtering / enhancement
    _a("common20_median__size3", "median_filter", {"size": 3}, "filtering", "moderate"),
    _a("common20_average__size3", "average_filter", {"size": 3}, "filtering", "moderate"),
    _a("common20_gaussian_blur__radius1", "gaussian_blur", {"radius": 1.0}, "filtering", "moderate"),
    _a("common20_sharpen__factor2", "sharpen", {"factor": 2.0}, "filtering", "moderate"),

    # Geometric / resampling
    _a("common20_rotation_back__deg1", "rotation_back", {"degrees": 1.0}, "geometric", "mild"),
    _a("common20_resize__factor0p75", "resize", {"factor": 0.75}, "geometric", "moderate"),
    _a("common20_crop_resize__keep0p9", "crop_resize", {"keep": 0.90}, "geometric", "moderate"),
    _a("common20_translation__x4_y4", "translation", {"shift_x": 4, "shift_y": 4}, "geometric", "mild"),

    # Photometric / point processing
    _a("common20_brightness__factor0p9", "brightness", {"factor": 0.9}, "photometric", "mild"),
    _a("common20_contrast__factor1p1", "contrast", {"factor": 1.1}, "photometric", "mild"),
    _a("common20_gamma__1p2", "gamma", {"gamma": 1.2}, "photometric", "moderate"),
    _a("common20_hist_equalization", "hist_equalization", {}, "photometric", "moderate"),
)

COMMON_20_GROUPS = {
    "compression_quantization": tuple(a for a in COMMON_20 if a.category == "compression"),
    "noise": tuple(a for a in COMMON_20 if a.category == "noise"),
    "filtering_enhancement": tuple(a for a in COMMON_20 if a.category == "filtering"),
    "geometric_resampling": tuple(a for a in COMMON_20 if a.category == "geometric"),
    "photometric_point_processing": tuple(a for a in COMMON_20 if a.category == "photometric"),
}

# The common suite is intentionally broad but excludes extreme destructive tests.
COMMON = _deduplicate(
    SANITY + COMPRESSION[:10] + NOISE[:9] + FILTERING[:9] + GEOMETRIC[:14] + PHOTOMETRIC[:11] + OCCLUSION[:4]
)
STRESS = _deduplicate(COMPRESSION + NOISE + FILTERING + GEOMETRIC + PHOTOMETRIC + OCCLUSION + COMBINED)

# Frozen suites used by the original proposal result scripts. These are kept
# byte-for-byte equivalent in parameterization so existing reported results can
# be reproduced; new experiments should use get_attack_suite("common") or
# get_attack_suite("stress").
LEGACY_MODERATE = (
    _a("jpeg_q90", "jpeg", {"quality": 90}, "compression", "mild"),
    _a("jpeg_q70", "jpeg", {"quality": 70}, "compression", "moderate"),
    _a("gaussian_noise_sigma1", "gaussian_noise", {"sigma": 1.0, "seed": 123}, "noise", "mild"),
    _a("salt_pepper_0p005", "salt_pepper", {"amount": 0.005, "seed": 123}, "noise", "moderate"),
    _a("median_3x3", "median_filter", {"size": 3}, "filtering", "moderate"),
    _a("gaussian_blur_r1", "gaussian_blur", {"radius": 1.0}, "filtering", "moderate"),
    _a("rotation_back_1deg", "rotation_back", {"degrees": 1.0}, "geometric", "mild"),
    _a("gamma_0p8", "gamma", {"gamma": 0.8}, "photometric", "moderate"),
    _a("gamma_1p2", "gamma", {"gamma": 1.2}, "photometric", "moderate"),
    _a("brightness_0p9", "brightness", {"factor": 0.9}, "photometric", "mild"),
    _a("brightness_1p1", "brightness", {"factor": 1.1}, "photometric", "mild"),
    _a("contrast_0p9", "contrast", {"factor": 0.9}, "photometric", "mild"),
    _a("contrast_1p1", "contrast", {"factor": 1.1}, "photometric", "mild"),
    _a("resize_0p75", "resize", {"factor": 0.75}, "geometric", "moderate"),
    _a("resize_1p5", "resize", {"factor": 1.5}, "geometric", "moderate"),
)

LEGACY_STRESS = (
    _a("jpeg_q50", "jpeg", {"quality": 50}, "compression", "strong"),
    _a("gaussian_noise_sigma5", "gaussian_noise", {"sigma": 5.0, "seed": 123}, "noise", "strong"),
    _a("gaussian_var_0p003", "gaussian_noise_var", {"variance": 0.003, "seed": 123}, "noise", "strong"),
    _a("speckle_0p01", "speckle_noise", {"variance": 0.01, "seed": 123}, "noise", "strong"),
    _a("median_5x5", "median_filter", {"size": 5}, "filtering", "strong"),
    _a("motion_blur_9", "motion_blur", {"size": 9}, "filtering", "strong"),
    _a("hist_equalization", "hist_equalization", {}, "photometric", "strong"),
    _a("rotation_2deg", "rotation", {"degrees": 2.0}, "geometric", "moderate"),
    _a("crop_resize_90", "crop_resize", {"keep": 0.90}, "geometric", "moderate"),
    _a("occlusion_25pct", "occlusion_fraction", {"fraction": 0.25, "position": "center"}, "occlusion", "strong"),
    _a("shear_0p08", "shear", {"shear_x": 0.08}, "geometric", "strong"),
    _a("jpeg70_plus_blur", "combined", {"steps": [
        {"group": "jpeg", "params": {"quality": 70}},
        {"group": "gaussian_blur", "params": {"radius": 1.0}},
    ]}, "combined", "strong"),
)


# Additional real-world and adversarial attacks.  Existing frozen suites remain
# unchanged so old result files are still reproducible.
REAL_WORLD = (
    _a("realworld_jpeg_recompress__q85_x2", "jpeg_recompression", {"quality": 85, "cycles": 2}, "compression", "moderate"),
    _a("realworld_jpeg_recompress__q70_x3", "jpeg_recompression", {"quality": 70, "cycles": 3}, "compression", "strong"),
    _a("realworld_screen_capture__default", "screen_capture", {"scale": 0.82, "gamma": 1.05, "jpeg_quality": 85}, "real_world", "strong"),
    _a("realworld_screen_capture__hard", "screen_capture", {"scale": 0.67, "gamma": 1.12, "jpeg_quality": 70}, "real_world", "extreme"),
    _a("realworld_print_scan__default", "print_scan", {"degrees": 0.4, "blur_radius": 0.6, "noise_sigma": 1.5, "jpeg_quality": 88, "seed": 123}, "real_world", "strong"),
    _a("realworld_print_scan__hard", "print_scan", {"degrees": 1.0, "blur_radius": 1.0, "noise_sigma": 3.0, "jpeg_quality": 75, "seed": 123}, "real_world", "extreme"),
    _a("realworld_grayscale", "grayscale", {}, "photometric", "moderate"),
    _a("realworld_channel_swap__bgr", "channel_permutation", {"order": [2, 1, 0]}, "photometric", "strong"),
    _a("realworld_dithering__colors64", "dithering", {"colors": 64}, "compression", "strong"),
    _a("realworld_copy_move__block64", "copy_move", {"block": 64, "seed": 123}, "tampering", "moderate"),
    _a("realworld_copy_move__block128", "copy_move", {"block": 128, "seed": 321}, "tampering", "strong"),
)

DEFORMATION = (
    _a("deformation_elastic__a1_s8", "elastic_warp", {"alpha": 1.0, "sigma": 8.0, "seed": 123}, "geometric", "moderate"),
    _a("deformation_elastic__a2_s8", "elastic_warp", {"alpha": 2.0, "sigma": 8.0, "seed": 123}, "geometric", "strong"),
    _a("deformation_elastic__a4_s10", "elastic_warp", {"alpha": 4.0, "sigma": 10.0, "seed": 123}, "geometric", "extreme"),
    _a("deformation_lens__barrel0p04", "lens_distortion", {"k1": 0.04}, "geometric", "moderate"),
    _a("deformation_lens__barrel0p08", "lens_distortion", {"k1": 0.08}, "geometric", "strong"),
    _a("deformation_lens__pincushion0p06", "lens_distortion", {"k1": -0.06}, "geometric", "strong"),
)

STRUCTURED_LOSS = (
    _a("structured_random_erasing__5pct_3", "random_erasing", {"fraction": 0.05, "rectangles": 3, "seed": 123}, "occlusion", "moderate"),
    _a("structured_random_erasing__15pct_5", "random_erasing", {"fraction": 0.15, "rectangles": 5, "seed": 123}, "occlusion", "strong"),
    _a("structured_random_erasing__30pct_8", "random_erasing", {"fraction": 0.30, "rectangles": 8, "seed": 123}, "occlusion", "extreme"),
    _a("structured_stripes__horizontal2_32", "stripe_dropout", {"orientation": "horizontal", "width": 2, "spacing": 32}, "occlusion", "moderate"),
    _a("structured_stripes__vertical2_32", "stripe_dropout", {"orientation": "vertical", "width": 2, "spacing": 32}, "occlusion", "moderate"),
    _a("structured_stripes__horizontal4_24", "stripe_dropout", {"orientation": "horizontal", "width": 4, "spacing": 24}, "occlusion", "strong"),
    _a("structured_stripes__vertical4_24", "stripe_dropout", {"orientation": "vertical", "width": 4, "spacing": 24}, "occlusion", "strong"),
)

EXTENDED_COMBINED = (
    _a("combined__jpeg50_gaussian3", "combined", {"steps": [
        {"group": "jpeg", "params": {"quality": 50}},
        {"group": "gaussian_noise", "params": {"sigma": 3.0, "seed": 123}},
    ]}, "combined", "extreme"),
    _a("combined__resize0p5_jpeg50", "combined", {"steps": [
        {"group": "resize", "params": {"factor": 0.5}},
        {"group": "jpeg", "params": {"quality": 50}},
    ]}, "combined", "extreme"),
    _a("combined__crop0p75_jpeg50", "combined", {"steps": [
        {"group": "crop_resize", "params": {"keep": 0.75}},
        {"group": "jpeg", "params": {"quality": 50}},
    ]}, "combined", "extreme"),
    _a("combined__rotation5_jpeg70_noise1", "combined", {"steps": [
        {"group": "rotation", "params": {"degrees": 5.0}},
        {"group": "jpeg", "params": {"quality": 70}},
        {"group": "gaussian_noise", "params": {"sigma": 1.0, "seed": 123}},
    ]}, "combined", "extreme"),
    _a("combined__grayscale_jpeg70", "combined", {"steps": [
        {"group": "grayscale", "params": {}},
        {"group": "jpeg", "params": {"quality": 70}},
    ]}, "combined", "strong"),
    _a("combined__randomerase15_jpeg70", "combined", {"steps": [
        {"group": "random_erasing", "params": {"fraction": 0.15, "rectangles": 5, "seed": 123}},
        {"group": "jpeg", "params": {"quality": 70}},
    ]}, "combined", "extreme"),
    _a("combined__lens0p04_jpeg80", "combined", {"steps": [
        {"group": "lens_distortion", "params": {"k1": 0.04}},
        {"group": "jpeg", "params": {"quality": 80}},
    ]}, "combined", "strong"),
    _a("combined__elastic1_jpeg80", "combined", {"steps": [
        {"group": "elastic_warp", "params": {"alpha": 1.0, "sigma": 8.0, "seed": 123}},
        {"group": "jpeg", "params": {"quality": 80}},
    ]}, "combined", "strong"),
)

# Balanced paper suite: all common attacks plus representative attacks from
# the newly added real-world, deformation, structured-loss and combined groups.
PUBLICATION = _deduplicate(
    COMMON
    + REAL_WORLD[:8]
    + DEFORMATION[:5]
    + STRUCTURED_LOSS[:5]
    + EXTENDED_COMBINED[:6]
)

# Largest deterministic suite for robustness exploration.  It intentionally
# includes severe and destructive cases and must not be summarized as one mean
# without category/severity breakdowns.
EXTENDED = _deduplicate(
    STRESS + REAL_WORLD + DEFORMATION + STRUCTURED_LOSS + EXTENDED_COMBINED
)

# Reviewer-driven, published-severity tests.  These presets are not used for
# tuning; results must be shown by group and severity, not pooled uncritically.
REVIEWER_STRENGTH = (
    tuple(_a(f'rev_jpeg_q{q}', 'jpeg', {'quality':q}, 'compression',
             'strong' if q <= 50 else 'moderate') for q in (90,70,50,40,30))
    + tuple(_a(f'rev_gaussian_var_{str(v).replace(".","p")}', 'gaussian_noise_var',
               {'variance': v,'seed':123}, 'noise', 'strong' if v >= 0.01 else 'moderate')
            for v in (0.003,0.01,0.05))
    + tuple(_a(f'rev_speckle_var_{str(v).replace(".","p")}', 'speckle_noise',
               {'variance':v,'seed':123}, 'noise', 'strong' if v >= 0.01 else 'moderate')
            for v in (0.001,0.01,0.03))
    + tuple(_a(f'rev_rotation_{str(deg).replace(".","p")}', 'rotation',
               {'degrees':deg}, 'geometric', 'strong' if deg >= 5 else 'moderate')
            for deg in (0.5,1.0,2.5,5.0,15.0))
    + tuple(_a(f'rev_rotation_back_{str(deg).replace(".","p")}', 'rotation_back',
               {'degrees':deg}, 'geometric_aligned', 'moderate')
            for deg in (1.0,2.5,5.0,15.0))
    + tuple(_a(f'rev_shift_subpixel_{str(d).replace(".","p")}', 'subpixel_translation',
               {'shift_x':d,'shift_y':d}, 'geometric', 'strong')
            for d in (0.25,0.5,0.75))
    + tuple(_a(f'rev_resize_{str(f).replace(".","p")}', 'resize',
               {'factor':f}, 'geometric', 'strong') for f in (0.75,0.9,1.1,1.25))
    + tuple(_a(f'rev_shear_{str(v).replace(".","p")}', 'shear',
               {'shear_x':v}, 'geometric', 'strong') for v in (0.03,0.08,0.15))
    + tuple(_a(f'rev_occlude_{int(fr*100)}pct', 'occlusion_fraction',
               {'fraction':fr}, 'occlusion', 'strong') for fr in (0.05,0.1,0.25,0.5))
    + tuple(_a(f'rev_carrier_attenuate_{str(a).replace(".","p")}', 'carrier_bandstop',
               {'attenuation':a}, 'intentional_removal', 'strong') for a in (0.0,0.25,0.5))
)
REVIEWER_GEOMETRIC = tuple(a for a in REVIEWER_STRENGTH if a.category.startswith('geometric'))
REVIEWER_SECURITY_ATTACKS = tuple(a for a in REVIEWER_STRENGTH if a.category == 'intentional_removal')

ATTACK_SUITES = {
    "sanity": SANITY,
    "compression": COMPRESSION,
    "noise": NOISE,
    "filtering": FILTERING,
    "geometric": GEOMETRIC,
    "photometric": PHOTOMETRIC,
    "occlusion": OCCLUSION,
    "combined": COMBINED,
    "common": COMMON,
    "common20": COMMON_20,
    "stress": STRESS,
    "real_world": REAL_WORLD,
    "deformation": DEFORMATION,
    "structured_loss": STRUCTURED_LOSS,
    "publication": PUBLICATION,
    "extended": EXTENDED,
    "reviewer_strength": REVIEWER_STRENGTH,
    "reviewer_geometric": REVIEWER_GEOMETRIC,
    "reviewer_security_attacks": REVIEWER_SECURITY_ATTACKS,
}


def get_attack_suite(name: str) -> tuple[AttackConfig, ...]:
    key = str(name).strip().lower()
    if key not in ATTACK_SUITES:
        raise KeyError(f"Unknown attack suite '{name}'. Valid: {', '.join(ATTACK_SUITES)}")
    return ATTACK_SUITES[key]


def list_attack_suites() -> dict[str, int]:
    return {name: len(configs) for name, configs in ATTACK_SUITES.items()}

def moderate_attacks() -> list[AttackConfig]:
    """Frozen original proposal suite retained for result reproducibility."""
    return list(LEGACY_MODERATE)

def stress_attacks() -> list[AttackConfig]:
    """Frozen original stress suite retained for result reproducibility."""
    return list(LEGACY_STRESS)

__all__ = [
    "ATTACK_SUITES", "COMMON_20", "COMMON_20_GROUPS", "get_attack_suite", "list_attack_suites",
    "moderate_attacks", "stress_attacks",
]
