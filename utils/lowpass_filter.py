# Low-pass filtering utility used by trajectory replay.

import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi

class RealTimeButterworthFilter:
    """
    Real-time second-order Butterworth filter.
    Suitable for filtering one sample at a time in a control loop.
    """
    
    def __init__(self, lowcut=None, highcut=None, fs=1000, btype='low', order=2, initial_value=0.0):
        """
        Initialize the filter.
        
        Args:
        lowcut: low cutoff frequency (Hz), for low-pass and high-pass filters
        highcut: high cutoff frequency (Hz), for band-pass and band-stop filters
        fs: sampling frequency (Hz)
        btype: filter type ('low', 'high', 'band', 'stop')
        order: filter order (default: 2)
        initial_value: initial value used to reduce startup transient response
        """
        self.fs = fs
        self.btype = btype
        self.order = order
        
        # Set cutoff frequencies according to filter type
        if btype == 'low':
            self.cutoff = lowcut
        elif btype == 'high':
            self.cutoff = lowcut
        elif btype in ['band', 'stop']:
            self.cutoff = [lowcut, highcut]
        else:
            raise ValueError("Filter type must be 'low', 'high', 'band', or 'stop'")
        
        # Design filter coefficients
        self.b, self.a = butter(self.order, self.cutoff, btype=btype, fs=fs)
        
        # Optimize initial state to reduce startup transient
        self.initial_value = initial_value
        self.zi = self._compute_initial_condition(initial_value)
        
        # Whether at least one sample has been processed
        self.is_initialized = False
        
        # Keep history for manual filtering path
        self.x_hist = np.zeros(len(self.b))
        self.y_hist = np.zeros(len(self.a) - 1)
    
    def _compute_initial_condition(self, initial_value):
        """Compute optimized initial conditions."""
        # Use SciPy to compute steady-state initial conditions
        zi = lfilter_zi(self.b, self.a)
        # Scale initial conditions by the provided initial value
        zi = zi * initial_value
        return zi
    
    def reset(self, initial_value=None):
        """Reset filter state."""
        if initial_value is not None:
            self.initial_value = initial_value
        
        self.zi = self._compute_initial_condition(self.initial_value)
        self.is_initialized = False
        
        # Reset history buffers
        self.x_hist = np.zeros(len(self.b))
        self.y_hist = np.zeros(len(self.a) - 1)
    
    def filter_step(self, x_new):
        """
        Single-step filtering with optimized initialization.
        
        Args:
        x_new: new input sample
        
        Returns:
        y_new: filtered output sample
        """
        # On first run, use the optimized initial condition
        if not self.is_initialized:
            # For the first sample, pre-warm using the configured initial value
            self.is_initialized = True
        
        # Use scipy lfilter and keep state continuity
        y_new, self.zi = lfilter(self.b, self.a, [x_new], zi=self.zi)
        
        return y_new[0]
    
    def filter_with_warmup(self, data, warmup_samples=10):
        """
        Filtering with a warmup phase, useful for initial segments.
        
        Args:
        data: input data array
        warmup_samples: number of warmup samples
        
        Returns:
        filtered_data: filtered data array
        """
        if len(data) == 0:
            return np.array([])
        
        # Save current state
        original_zi = self.zi.copy()
        original_initialized = self.is_initialized
        
        # Reset to initial value
        self.reset(initial_value=data[0] if len(data) > 0 else 0.0)
        
        # Warmup phase
        warmup_data = np.full(warmup_samples, data[0])
        for sample in warmup_data:
            self.filter_step(sample)
        
        # Filter actual data
        filtered_data = []
        for sample in data:
            filtered_sample = self.filter_step(sample)
            filtered_data.append(filtered_sample)
        
        # Restore original state
        self.zi = original_zi
        self.is_initialized = original_initialized
        
        return np.array(filtered_data)
    
    def _manual_filter_with_initialization(self, x_new):
        """
        Manually implemented filtering path with optimized initialization.
        """
        if not self.is_initialized:
            # Initialize history buffers
            self.x_hist.fill(x_new)
            self.y_hist.fill(x_new)
            self.is_initialized = True
        
        # Update input history
        self.x_hist = np.roll(self.x_hist, 1)
        self.x_hist[0] = x_new
        
        # Compute output
        y_new = (np.dot(self.b, self.x_hist) - np.dot(self.a[1:], self.y_hist)) / self.a[0]
        
        # Update output history
        self.y_hist = np.roll(self.y_hist, 1)
        self.y_hist[0] = y_new
        
        return y_new
    
    def get_filter_info(self):
        """Get filter metadata."""
        info = {
            'type': f'{self.order}-order Butterworth {self.btype} filter',
            'sampling_frequency': self.fs,
            'cutoff_frequencies': self.cutoff,
            'numerator_coeffs': self.b,
            'denominator_coeffs': self.a,
            'initial_value': self.initial_value,
            'is_initialized': self.is_initialized
        }
        return info
