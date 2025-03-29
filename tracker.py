import torch
import torch.profiler
import csv
import os
from typing import Callable
from functools import wraps

def profile_training_flops_by_steps(csv_path="flops_per_step.csv", log_every_n_steps=10):
    """
    A decorator/wrapper function to profile FLOPs during model training,
    logging results every N steps to a CSV file.
    
    Args:
        csv_path (str): Path to save the CSV log file
        log_every_n_steps (int): How frequently to log FLOP measurements
        
    Returns:
        A decorator function that profiles the training process
    """
    def decorator(training_func: Callable):
        @wraps(training_func)
        def wrapper(*args, **kwargs):
            # Get the trainer from args (assuming first arg is trainer)
            trainer = args[0]
            original_training_step = trainer.training_step
            
            # Create CSV file and write header
            os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
            with open(csv_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['Step', 'FLOPs', 'CUDA Time (ms)', 'CPU Time (ms)', 'Memory (MB)'])
            
            # Step counter
            step_counter = [0]  # Using list for nonlocal access
            
            # Override the training_step method to add profiling
            def profiled_training_step(*step_args, **step_kwargs):
                step_counter[0] += 1
                current_step = step_counter[0]
                
                # Only profile every N steps
                if current_step % log_every_n_steps == 0:
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA
                        ],
                        record_shapes=True,
                        profile_memory=True,
                        with_flops=True
                    ) as prof:
                        result = original_training_step(*step_args, **step_kwargs)
                    
                    # Calculate total FLOPs
                    total_flops = 0
                    total_cuda_time_ms = 0
                    total_cpu_time_ms = 0
                    
                    for event in prof.key_averages():
                        # Accumulate FLOPs
                        if hasattr(event, 'flops') and event.flops > 0:
                            total_flops += event.flops
                        
                        # Accumulate CUDA time (converted to ms)
                        if hasattr(event, 'cuda_time'):
                            total_cuda_time_ms += event.cuda_time * 1000  # Convert to ms
                        
                        # Accumulate CPU time (converted to ms)
                        if hasattr(event, 'cpu_time'):
                            total_cpu_time_ms += event.cpu_time * 1000  # Convert to ms
                    
                    # Calculate memory usage in MB
                    memory_usage_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                    
                    # Log to CSV
                    with open(csv_path, 'a', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow([
                            current_step, 
                            total_flops, 
                            total_cuda_time_ms, 
                            total_cpu_time_ms,
                            memory_usage_mb
                        ])
                    
                    # Print to console
                    print(f"\nStep {current_step}:")
                    print(f"  Total FLOPs: {total_flops:,}")
                    print(f"  CUDA Time: {total_cuda_time_ms:.2f} ms")
                    print(f"  CPU Time: {total_cpu_time_ms:.2f} ms")
                    print(f"  Memory Usage: {memory_usage_mb:.2f} MB")
                    
                    # Reset peak memory stats for next iteration
                    torch.cuda.reset_peak_memory_stats()
                    
                    return result
                else:
                    # Run without profiling for other steps
                    return original_training_step(*step_args, **step_kwargs)
            
            # Replace the training_step method
            trainer.training_step = profiled_training_step
            
            try:
                # Execute the original training function
                result = training_func(*args, **kwargs)
                return result
            finally:
                # Restore original training_step method
                trainer.training_step = original_training_step
                print(f"\nFLOP measurements have been saved to {csv_path}")
        
        return wrapper
    return decorator