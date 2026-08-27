import numpy as np

from bookshelf_simple_experiment_ros.geometry import (
    REVIEWED_EEF_BOOK_QUATERNION_XYZW,
    REVIEWED_EEF_BOOK_TRANSLATION_XYZ,
    compute_preinsert_target,
    invert_transform,
    make_transform,
    reviewed_eef_book_transform,
)


def test_reviewed_transform_values_and_positive_signs_are_exact():
    assert REVIEWED_EEF_BOOK_TRANSLATION_XYZ == (
        0.006189808263520789,
        0.004397635899244547,
        0.18076520526773382,
    )
    assert REVIEWED_EEF_BOOK_QUATERNION_XYZW == (
        0.7170947434170492,
        0.01281329455160485,
        0.6961397093730864,
        0.03162994594249451,
    )


def test_preinsert_composition_recovers_requested_book_pose():
    base_slot = make_transform([0.8, 0.1, 0.2], [0.0, 0.0, 0.0, 1.0])
    eef_book = reviewed_eef_book_transform()
    eef_tcp = make_transform([0.0, 0.0, 0.107])
    target = compute_preinsert_target(base_slot, eef_book, eef_tcp)
    np.testing.assert_allclose(
        target.transform_slot_book[:3, 3], [-0.108, 0.0, 0.006], atol=1.0e-12
    )
    np.testing.assert_allclose(
        target.transform_base_eef @ eef_book,
        target.transform_base_book,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        target.transform_base_eef @ eef_tcp,
        target.transform_base_tcp,
        atol=1.0e-12,
    )


def test_transform_inverse_is_rigid_identity():
    transform = reviewed_eef_book_transform()
    np.testing.assert_allclose(transform @ invert_transform(transform), np.eye(4), atol=1e-12)
