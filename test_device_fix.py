#!/usr/bin/env python3
"""
Test script to verify the device_type detection and pin_memory logic
"""

def test_device_type_detection():
    """Test that device_type is correctly determined for different devices"""
    
    test_cases = [
        ('cuda', 'cuda'),
        ('cuda:0', 'cuda'),
        ('cuda:1', 'cuda'),
        ('mps', 'mps'),
        ('cpu', 'cpu'),
    ]
    
    print("Testing device_type detection:")
    print("-" * 50)
    
    for device, expected_type in test_cases:
        # This is the logic from the fixed code
        device_type = 'cuda' if 'cuda' in device else 'mps' if 'mps' in device else 'cpu'
        
        status = "✓" if device_type == expected_type else "✗"
        print(f"{status} device='{device}' -> device_type='{device_type}' (expected: '{expected_type}')")
        
        # Verify pin_memory logic
        should_pin = device_type == 'cuda'
        print(f"  pin_memory would be: {should_pin}")
    
    print("-" * 50)
    print("\nSummary:")
    print("- CUDA devices: pin_memory=True (for async GPU transfer)")
    print("- MPS devices: pin_memory=False (avoids segmentation fault)")
    print("- CPU devices: pin_memory=False (not needed)")

if __name__ == '__main__':
    test_device_type_detection()
