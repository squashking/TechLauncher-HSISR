import sys
import os

import numpy as np
import spectral.io.envi as envi


def calibration(dark_hsi, ref_hsi, input_hsi, output_filename):

    assert dark_hsi.shape == ref_hsi.shape

    subt = np.repeat(
        np.average(dark_hsi[:, :, :], axis=0, keepdims=True),
        input_hsi.shape[0],
        axis=0
    )
    divisor = np.repeat(
        np.average(ref_hsi[:, :, :] - dark_hsi[:, :, :], axis=0, keepdims=True),
        input_hsi.shape[0],
        axis=0
    )

    result = np.clip(
        ((input_hsi[:, :, :] - subt) / divisor) * 255,
        0,
        255
    )

    if output_filename is None:
        output_hdr = "calibration.hdr"
        output_bil = "calibration.bil"
    else:
        output_hdr = output_filename + ".hdr"
        output_bil = output_filename + ".bil"

    envi.save_image(output_hdr, result, dtype=np.uint16, interleave="bil", ext="bil", force=True, metadata=dark_hsi.metadata)
    output_hsi = envi.open(output_hdr, output_bil)

    #if output_filename is None:
    #    os.remove(output_hdr)
    #    os.remove(output_bil)

    return output_hsi
