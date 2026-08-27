"""Stack shots into one captioned figure for the PR discussion."""

import sys

from PIL import Image, ImageDraw, ImageFont

S = "D:/meta-mne-python-sprint/validation/browser_fuzz/shots/"
FONT = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 17)
BOLD = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 18)


def stack(out, panels, width=900, pad=10, cap_h=52):
    ims = []
    for path, title, sub in panels:
        im = Image.open(S + path).convert("RGB")
        if im.width != width:
            im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
        ims.append((im, title, sub))
    H = sum(im.height + cap_h + pad for im, _, _ in ims) + pad
    canvas = Image.new("RGB", (width + 2 * pad, H), "white")
    d = ImageDraw.Draw(canvas)
    y = pad
    for im, title, sub in ims:
        d.text((pad, y), title, fill="black", font=BOLD)
        d.text((pad, y + 24), sub, fill=(70, 70, 70), font=FONT)
        y += cap_h
        canvas.paste(im, (pad, y))
        d.rectangle([pad, y, pad + im.width - 1, y + im.height - 1], outline=(190, 190, 190))
        y += im.height + pad
    canvas.save(S + out)
    print(S + out, canvas.size)


stack(
    "a8_FIGURE_f1_vline_clamped.png",
    [
        (
            "a8_f1_ragged_1_before_scroll.png",
            "1. Qt, epochs of 100/250/75/180 samples @100 Hz, vline placed at latency 0.900 s",
            "Both lines are at 0.900 s in their own epoch. Correct.",
        ),
        (
            "a8_f1_ragged_2_after_scroll.png",
            "2. Qt, after one scroll right (epochs 1 and 2). Epoch 2 is only 0.75 s long.",
            "The second line is CLAMPED to the epoch's last sample, reading 0.740 s. It should be hidden.",
        ),
        (
            "a8_mplvline_ragged_2_after_scroll.png",
            "3. matplotlib, same data, same gesture",
            "One line, in epoch 1 only. The short epoch correctly gets none. This is what Qt should do.",
        ),
        (
            "a8_f1_fixed_2_after_scroll.png",
            "4. Qt control: four equal 150-sample epochs, same gesture",
            "Both lines at 0.900 s. The fixed-duration path is unaffected.",
        ),
    ],
)

stack(
    "a8_FIGURE_overview_click_blank.png",
    [
        (
            "a8_ovclick_ragged_0_open.png",
            "1. Qt, epochs of 100/250/75/180 samples, n_epochs=2, at open",
            "View [0.000, 3.500] = epochs 0-1 exactly. Axis reads 0 1.",
        ),
        (
            "a8_ovclick_ragged_1_click_far_right.png",
            "2. After one click at the right end of the overview bar",
            "View [2.550, 6.050]; only epochs 2-3 ([3.500, 6.040]) are loaded. 27.4% blank, axis still reads 2 3.",
        ),
        (
            "a8_ovclick_fixed_1_click_far_right.png",
            "3. Control: four equal 150-sample epochs, same click",
            "View [3.000, 6.000] = epochs 2-3 exactly. Nothing blank.",
        ),
    ],
)

stack(
    "a8_FIGURE_missing_epoch_labels.png",
    [
        (
            "a8_ticks_short_first.png",
            "1. Qt, epochs of 3/300/300/300 samples. get_labels() reports ['0','1','2','3'].",
            "Only 1 2 3 are painted; epoch 0's number is dropped at the axis edge.",
        ),
        (
            "a8_ticks_short_last.png",
            "2. Qt, epochs of 300/300/300/3 samples. get_labels() reports ['0','1','2','3'].",
            "Only 0 1 2 are painted; the short last epoch's number is dropped at the right edge.",
        ),
        (
            "a8_bounds_equal_control_900px.png",
            "3. Control: four equal 150-sample epochs",
            "All four numbers painted, at every width tested (300-1600 px).",
        ),
    ],
)
