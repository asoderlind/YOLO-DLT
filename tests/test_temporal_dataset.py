from collections import defaultdict
from unittest.mock import MagicMock, patch

import pytest
import torch

from ultralytics.data.dataset import TemporalYOLODataset


class TestTemporalDataset:
    @pytest.fixture
    def mock_image_files(self):
        """Create mock image files with proper naming convention."""
        return [
            "path/to/vid0_frame0_image_0.jpg",
            "path/to/vid0_frame1_image_0.jpg",
            "path/to/vid0_frame2_image_0.jpg",
            "path/to/vid0_frame3_image_0.jpg",
            "path/to/vid0_frame4_image_0.jpg",
            "path/to/vid1_frame0_image_0.jpg",
            "path/to/vid1_frame1_image_0.jpg",
            "path/to/vid1_frame2_image_0.jpg",
        ]

    @pytest.fixture
    def mock_dataset(self, mock_image_files):
        """Create a mocked TemporalYOLODataset instance."""
        # Create a more extensive mock setup with proper data attribute
        with (
            patch("ultralytics.data.dataset.TemporalYOLODataset._build_video_index"),
            patch("ultralytics.data.dataset.TemporalYOLODataset.get_img_files", return_value=mock_image_files),
            patch("ultralytics.data.dataset.TemporalYOLODataset.get_labels"),
            patch("ultralytics.data.dataset.TemporalYOLODataset.update_labels"),
            patch("ultralytics.data.dataset.BaseDataset.build_transforms"),
            patch("ultralytics.data.augment.v8_transforms") as mock_transforms,
        ):
            # Create mock data with required attributes
            mock_data = MagicMock()
            mock_data.get.return_value = []  # For flip_idx

            # Create mock hyperparameters
            hyp = MagicMock()
            hyp.get.side_effect = lambda key, default: {"temporal_window": 3, "temporal_stride": 1}.get(key, default)

            # Initialize dataset with mocked data
            dataset = TemporalYOLODataset(
                img_path="dummy_path",
                imgsz=640,
                hyp=hyp,
                data=mock_data,  # Pass mock data
            )

            # Set up transforms to avoid further errors
            mock_transforms.return_value = MagicMock()
            dataset.transforms = MagicMock()

            # Mock internal structures
            dataset.im_files = mock_image_files
            dataset.dataset_idx_to_video_id = {
                0: 0,
                1: 0,
                2: 0,
                3: 0,
                4: 0,  # vid0
                5: 1,
                6: 1,
                7: 1,  # vid1
            }
            dataset.video_to_frames = defaultdict(list)
            dataset.video_to_frames[0] = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
            dataset.video_to_frames[1] = [(0, 5), (1, 6), (2, 7)]

            # Mock __getitem__ to return simpler data for testing
            def mock_getitem(self, index):
                return {"img": torch.rand(3, 64, 64), "batch_idx": torch.tensor([0])}

            # Patch YOLODataset.__getitem__ to avoid dependency on actual data loading
            with patch("ultralytics.data.dataset.YOLODataset.__getitem__", mock_getitem):
                yield dataset

    def test_parse_path(self, mock_dataset):
        """Test the path parsing functionality."""
        path = "path/to/vid5_frame10_image_0.jpg"
        vid_id, frame_id = mock_dataset._parse_path(path)

        assert vid_id == 5
        assert frame_id == 10

    def test_parse_path_invalid(self, mock_dataset):
        """Test that invalid paths raise appropriate errors."""
        with pytest.raises(ValueError):
            mock_dataset._parse_path("invalid_filename.jpg")

    @pytest.mark.parametrize(
        "idx,expected_frames",
        [
            (2, [(1, 1), (0, 0)]),  # Frame 2, should get frames 1 and 0
            (4, [(3, 3), (2, 2), (1, 1)]),  # Frame 4, should get frames 3, 2, and 1
            (0, []),  # Frame 0, should get no previous frames
        ],
    )
    def test_get_reference_frames_local(self, mock_dataset, idx, expected_frames):
        """Test getting reference frames with local sampling."""
        ref_frames = mock_dataset.get_reference_frames(idx, global_sampling=False)
        assert ref_frames == expected_frames

    def test_get_reference_frames_global(self, mock_dataset):
        """Test getting reference frames with global sampling."""
        # For global sampling, we expect random frames, so we just check the count
        ref_frames = mock_dataset.get_reference_frames(2, global_sampling=True)
        assert len(ref_frames) == min(3, len(mock_dataset.video_to_frames[0]))

        # Ensure all frames are from the same video
        for _, idx in ref_frames:
            assert mock_dataset.dataset_idx_to_video_id[idx] == 0

    def test_insufficient_frames_for_window(self, mock_dataset):
        """Test behavior when there aren't enough frames to fill the temporal window."""
        # Set a larger temporal window
        mock_dataset.temporal_window = 8

        # Test with a frame that doesn't have enough previous frames
        ref_frames = mock_dataset.get_reference_frames(4, global_sampling=False)

        # Should only get as many frames as available (4, 3, 2, 1, 0)
        # But frame 4 itself is excluded because we're looking at previous frames
        assert len(ref_frames) == 4
        assert ref_frames == [(3, 3), (2, 2), (1, 1), (0, 0)]

    def test_getitem_return_structure(self, mock_dataset):
        """Test the structure of data returned by __getitem__."""
        # Since we've mocked __getitem__, we can directly test our implementation
        main_data, ref_data = mock_dataset.__getitem__(3)

        # Verify main data structure
        assert isinstance(main_data, dict)
        assert "img" in main_data
        assert "batch_idx" in main_data

        # Verify reference data structure
        assert isinstance(ref_data, list)

    @pytest.mark.parametrize(
        "idx,expected_ref_count",
        [
            (0, 0),  # Frame 0 should have no reference frames
            (3, 3),  # Frame 3 should have 3 reference frames (0, 1, 2)
            (4, 3),  # Frame 4 should have 3 reference frames (1, 2, 3)
        ],
    )
    def test_reference_frame_count(self, mock_dataset, idx, expected_ref_count):
        """Test that the correct number of reference frames is returned."""
        # Reset temporal window to default for this test
        mock_dataset.temporal_window = 3
        mock_dataset.temporal_stride = 1

        # Get reference frames
        _, ref_data = mock_dataset.__getitem__(idx)

        # Verify count
        assert len(ref_data) == expected_ref_count

    def test_collate_fn(self):
        """Test the collate_fn method with mocked batch data."""
        # Create mock batch data
        batch = [
            (
                {"img": torch.rand(3, 64, 64), "batch_idx": torch.tensor([0])},
                [
                    {"img": torch.rand(3, 64, 64), "batch_idx": torch.tensor([0])},
                    {"img": torch.rand(3, 64, 64), "batch_idx": torch.tensor([0])},
                ],
            ),
            (
                {"img": torch.rand(3, 64, 64), "batch_idx": torch.tensor([1])},
                [{"img": torch.rand(3, 64, 64), "batch_idx": torch.tensor([1])}],
            ),
        ]

        # Mock YOLODataset.collate_fn to return a simplified dictionary
        with patch(
            "ultralytics.data.dataset.YOLODataset.collate_fn",
            side_effect=lambda x: {
                "img": torch.stack([i["img"] for i in x]),
                "batch_idx": torch.cat([i["batch_idx"] for i in x]),
            },
        ):
            result = TemporalYOLODataset.collate_fn(batch)

            # Verify result structure
            assert "img" in result
            assert "batch_idx" in result
            assert "reference_frames" in result

            # Verify reference frames structure
            assert len(result["reference_frames"]) == 2
            assert result["reference_frames"][0] is not None
            assert result["reference_frames"][1] is not None

    def test_temporal_stride(self, mock_dataset):
        """Test that temporal stride is correctly applied."""
        # Set larger stride
        mock_dataset.temporal_stride = 2
        mock_dataset.temporal_window = 3

        # For frame 4, with stride 2, we should get frames 2 and 0
        ref_frames = mock_dataset.get_reference_frames(4, global_sampling=False)
        expected = [(2, 2), (0, 0)]

        assert ref_frames == expected
