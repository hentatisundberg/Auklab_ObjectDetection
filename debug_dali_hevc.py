#!/usr/bin/env python3
"""
Debug script to test DALI with HEVC video
"""

import nvidia.dali as dali
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
import nvidia.dali.types as types

@pipeline_def(batch_size=1, num_threads=2, device_id=0)
def minimal_video_pipeline(video_path):
    """Minimal DALI pipeline for HEVC debugging"""
    
    # Try the most basic video reading
    video = fn.readers.video(
        device="gpu",
        file_root="",
        filenames=[video_path],
        sequence_length=1,  # Single frame
        step=1,
        stride=1,
        normalized=False,
        random_shuffle=False,
        pad_last_batch=True,
        skip_vfr_check=True,
        name="video_reader"
    )
    
    return video

def test_hevc_video():
    """Test HEVC video with minimal DALI configuration"""
    video_path = "vid/input2.mkv"
    
    print(f"🧪 Testing HEVC video: {video_path}")
    
    try:
        print("   📝 Creating minimal pipeline...")
        pipeline = minimal_video_pipeline(video_path)
        
        print("   🔧 Building pipeline...")
        pipeline.build()
        
        print("   🎬 Creating iterator...")
        iterator = DALIGenericIterator(
            [pipeline],
            ['video'],
            reader_name='video_reader',
            last_batch_policy=LastBatchPolicy.FILL,
            auto_reset=True
        )
        
        print("   🎯 Testing first batch...")
        batch = next(iterator)
        video_data = batch[0]['video']
        
        print(f"   ✅ Success! Video shape: {video_data.shape}")
        print(f"   📊 Data type: {video_data.dtype}")
        
        # Try a few more batches
        for i in range(3):
            batch = next(iterator)
            print(f"   ✅ Batch {i+2} successful")
            
        return True
        
    except Exception as e:
        print(f"   ❌ HEVC test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_progressive_complexity():
    """Test with increasing complexity"""
    video_path = "vid/input2.mkv"
    
    configs = [
        {"seq_len": 1, "step": 1, "batch": 1},
        {"seq_len": 1, "step": 25, "batch": 1}, 
        {"seq_len": 4, "step": 25, "batch": 1},
        {"seq_len": 8, "step": 25, "batch": 1},
    ]
    
    for config in configs:
        print(f"\n🧪 Testing config: {config}")
        
        @pipeline_def(batch_size=config["batch"], num_threads=2, device_id=0)
        def test_pipeline():
            video = fn.readers.video(
                device="gpu",
                file_root="",
                filenames=[video_path],
                sequence_length=config["seq_len"],
                step=config["step"],
                stride=1,
                normalized=False,
                random_shuffle=False,
                pad_last_batch=True,
                skip_vfr_check=True,
                name="video_reader"
            )
            return video
        
        try:
            pipeline = test_pipeline()
            pipeline.build()
            
            iterator = DALIGenericIterator(
                [pipeline],
                ['video'],
                reader_name='video_reader',
                last_batch_policy=LastBatchPolicy.FILL,
                auto_reset=True
            )
            
            batch = next(iterator)
            print(f"   ✅ Config {config} successful! Shape: {batch[0]['video'].shape}")
            
        except Exception as e:
            print(f"   ❌ Config {config} failed: {e}")
            break

if __name__ == "__main__":
    print("🔍 HEVC/H.265 Video Debug Test")
    print("=" * 50)
    
    success = test_hevc_video()
    
    if success:
        print("\n" + "=" * 50)
        test_with_progressive_complexity()
    else:
        print("\n💡 HEVC compatibility issue detected!")
        print("   Possible solutions:")
        print("   1. Convert video to H.264")
        print("   2. Use different DALI decoder settings")
        print("   3. Fall back to OpenCV decoding")